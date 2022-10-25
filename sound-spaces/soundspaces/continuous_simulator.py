# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from turtle import distance, pos, position
from typing import Any, List, Optional
from abc import ABC
from collections import defaultdict, namedtuple
import logging
import time
import pickle
import os
import json
import itertools

import librosa
import psutil
import scipy
from scipy.io import wavfile
from scipy.signal import fftconvolve
import numpy as np
import networkx as nx
from gym import spaces

from habitat.core.registry import registry
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis, quat_from_coeffs, quat_to_angle_axis
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.sims.habitat_simulator.habitat_simulator import HabitatSimSensor, overwrite_config
from habitat.core.simulator import (
    AgentState,
    Config,
    Observations,
    SensorSuite,
    ShortestPathPoint,
    Simulator,
)
from soundspaces.utils import load_metadata
from soundspaces.mp3d_utils import HouseReader


def calculate_mem_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024.0 / 1024 / 1024  # in GB


def crossfade(x1, x2, sr):
    crossfade_samples = int(0.05 * sr)  # 30 ms
    x2_weight = np.arange(crossfade_samples + 1) / crossfade_samples
    x1_weight = np.flip(x2_weight)
    x3 = [x1[:, :crossfade_samples+1] * x1_weight + x2[:, :crossfade_samples+1] * x2_weight, x2[:, crossfade_samples+1:]]

    return np.concatenate(x3, axis=1)


@registry.register_simulator()
class ContinuousSoundSpacesSim(Simulator, ABC):
    r"""Changes made to simulator wrapper over habitat-sim

    This simulator first loads the graph of current environment and moves the agent among nodes.
    Any sounds can be specified in the episode and loaded in this simulator.
    Args:
        config: configuration for initializing the simulator.
    """

    def action_space_shortest_path(self, source: AgentState, targets: List[AgentState], agent_id: int = 0) -> List[
            ShortestPathPoint]:
        pass

    def __init__(self, config: Config) -> None:
        self.config = self.habitat_config = config
        agent_config = self._get_agent_config()
        sim_sensors = []
        for sensor_name in agent_config.SENSORS:
            sensor_cfg = getattr(self.config, sensor_name)
            sensor_type = registry.get_sensor(sensor_cfg.TYPE)

            assert sensor_type is not None, "invalid sensor type {}".format(
                sensor_cfg.TYPE
            )
            sim_sensors.append(sensor_type(sensor_cfg))

        self._sensor_suite = SensorSuite(sim_sensors)
        self.sim_config = self.create_sim_config(self._sensor_suite)
        self._current_scene = self.sim_config.sim_cfg.scene_id
        self._action_space = spaces.Discrete(
            len(self.sim_config.agents[0].action_space)
        )
        self._prev_sim_obs = None

        self._source_position_index = None
        self._receiver_position_index = None
        self._rotation_angle = None
        self._current_sounds = None
        self._offset = None
        self._duration = None
        self._audio_index = None
        self._audio_length = None
        self._source_sound_dict = dict()
        self._sampling_rate = None
        self._node2index = None
        self._scene_observations = None
        self._episode_step_count = None
        self._is_episode_active = None
        self._position_to_index_mapping = dict()
        self._previous_step_collided = False
        self._house_readers = dict()

        self.points, self.graph = load_metadata(self.metadata_dir)
        self._sim = habitat_sim.Simulator(config=self.sim_config)
        self._last_rir = None
        self._current_sample_index = 0
        self._audio_sensor_num = self.config.AUDIO.NUM
        self.add_acoustic_config()

    def add_acoustic_config(self):
        """
        シミュレータ(self._sim)に、audio_sensorを加える
        デフォルトではカメラしかいれていないっぽいのでここで新しく追加する必要がある
        """
        import pprint
        for i in range(self._audio_sensor_num):
            acoustics_config = habitat_sim.sensor.RLRAudioPropagationConfiguration()
            acoustics_config.threadCount = 3
            acoustics_config.sampleRate = self.config.AUDIO.RIR_SAMPLING_RATE
            acoustics_config.irTime = self.config.AUDIO.IR_TIME
            acoustics_config.indirectRayCount = 500
            acoustics_config.temporalCoherence = True
            acoustics_config.transmission = True
            acoustics_config.enableMaterials = False

            channel_layout = habitat_sim.sensor.RLRAudioPropagationChannelLayout()
            channel_layout.channelType = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Binaural
            # channel_layout.channelType = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Ambisonics
            # channel_layout.channelType = habitat_sim.sensor.RLRAudioPropagationChannelLayoutType.Quad
            channel_layout.channelCount = 2

            audio_sensor_spec = habitat_sim.AudioSensorSpec()
            audio_sensor_spec.uuid = f"audio_sensor_{i}"
            # audio_sensor_spec.uuid = f"audio_sensor"
            audio_sensor_spec.acousticsConfig = acoustics_config
            audio_sensor_spec.channelLayout = channel_layout

            # f = open("debug.txt", "a")
            # f.write(f"acousticsConfig:\n")
            # pprint.pprint(acoustics_config, f)
            # f.write(f"channelLayout\n")
            # pprint.pprint(channel_layout, f)
            # f.close()

            self._sim.add_sensor(audio_sensor_spec)

    def create_sim_config(
        self, _sensor_suite: SensorSuite
    ) -> habitat_sim.Configuration:
        sim_config = habitat_sim.SimulatorConfiguration()
        # Check if Habitat-Sim is post Scene Config Update
        if not hasattr(sim_config, "scene_id"):
            raise RuntimeError(
                "Incompatible version of Habitat-Sim detected, please upgrade habitat_sim"
            )
        overwrite_config(
            config_from=self.config.HABITAT_SIM_V0,
            config_to=sim_config,
            # Ignore key as it gets propogated to sensor below
            ignore_keys={"gpu_gpu"},
        )
        sim_config.scene_id = self.config.SCENE
        sim_config.enable_physics = False
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/mp3d/mp3d.scene_dataset_config.json'
        # sim_config.scene_dataset_config_file = 'data/scene_datasets/replica/replica.scene_dataset_config.json'
        agent_config = habitat_sim.AgentConfiguration()
        overwrite_config(
            config_from=self._get_agent_config(),
            config_to=agent_config,
            # These keys are only used by Hab-Lab
            ignore_keys={
                "is_set_start_state",
                # This is the Sensor Config. Unpacked below
                "sensors",
                "start_position",
                "start_rotation",
                "goal_positions",
                "offset",
                "duration",
                "sound_ids",
                "mass",
                "linear_acceleration",
                "angular_acceleration",
                "linear_friction",
                "angular_friction",
                "coefficient_of_restitution",
                "distractor_sound_id",
                "distractor_position_index"
            },
        )

        sensor_specifications = []
        for sensor in _sensor_suite.sensors.values():
            assert isinstance(sensor, HabitatSimSensor)
            sim_sensor_cfg = sensor._get_default_spec()  # type: ignore[misc]
            overwrite_config(
                config_from=sensor.config,
                config_to=sim_sensor_cfg,
                # These keys are only used by Hab-Lab
                # or translated into the sensor config manually
                ignore_keys=sensor._config_ignore_keys,
                # TODO consider making trans_dict a sensor class var too.
                trans_dict={
                    "sensor_model_type": lambda v: getattr(
                        habitat_sim.FisheyeSensorModelType, v
                    ),
                    "sensor_subtype": lambda v: getattr(
                        habitat_sim.SensorSubType, v
                    ),
                },
            )
            sim_sensor_cfg.uuid = sensor.uuid
            sim_sensor_cfg.resolution = list(
                sensor.observation_space.shape[:2]
            )

            # TODO(maksymets): Add configure method to Sensor API to avoid
            # accessing child attributes through parent interface
            # We know that the Sensor has to be one of these Sensors
            sim_sensor_cfg.sensor_type = sensor.sim_sensor_type
            sim_sensor_cfg.gpu2gpu_transfer = (
                self.config.HABITAT_SIM_V0.GPU_GPU
            )
            sensor_specifications.append(sim_sensor_cfg)

        agent_config.sensor_specifications = sensor_specifications
        agent_config.action_space = registry.get_action_space_configuration(
            self.config.ACTION_SPACE_CONFIG
        )(self.config).get()

        return habitat_sim.Configuration(sim_config, [agent_config])

    @property
    def sensor_suite(self) -> SensorSuite:
        return self._sensor_suite

    def _update_agents_state(self) -> bool:
        is_updated = False
        for agent_id, _ in enumerate(self.config.AGENTS):
            agent_cfg = self._get_agent_config(agent_id)
            if agent_cfg.IS_SET_START_STATE:
                self.set_agent_state(
                    agent_cfg.START_POSITION,
                    agent_cfg.START_ROTATION,
                    agent_id,
                )
                is_updated = True

        return is_updated

    def _get_agent_config(self, agent_id: Optional[int] = None) -> Any:
        if agent_id is None:
            agent_id = self.config.DEFAULT_AGENT_ID
        agent_name = self.config.AGENTS[agent_id]
        agent_config = getattr(self.config, agent_name)
        return agent_config

    def get_agent_state(self, agent_id: int = 0) -> habitat_sim.AgentState:
        return self._sim.get_agent(agent_id).get_state()

    def set_agent_state(
        self,
        position: List[float],
        rotation: List[float],
        agent_id: int = 0,
        reset_sensors: bool = True,
    ) -> bool:
        agent = self._sim.get_agent(agent_id)
        new_state = self.get_agent_state(agent_id)
        new_state.position = position
        new_state.rotation = rotation

        # NB: The agent state also contains the sensor states in _absolute_
        # coordinates. In order to set the agent's body to a specific
        # location and have the sensors follow, we must not provide any
        # state for the sensors. This will cause them to follow the agent's
        # body
        new_state.sensor_states = {}
        agent.set_state(new_state, reset_sensors)
        return True

    @property
    def binaural_rir_dir(self):
        return os.path.join(self.config.AUDIO.BINAURAL_RIR_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def source_sound_dir(self):
        return self.config.AUDIO.SOURCE_SOUND_DIR

    @property
    def distractor_sound_dir(self):
        return self.config.AUDIO.DISTRACTOR_SOUND_DIR

    @property
    def metadata_dir(self):
        return os.path.join(self.config.AUDIO.METADATA_DIR, self.config.SCENE_DATASET, self.current_scene_name)

    @property
    def current_scene_name(self):
        # config.SCENE (_current_scene) looks like 'data/scene_datasets/replica/office_1/habitat/mesh_semantic.ply'
        return self._current_scene.split('/')[3]

    @property
    def current_scene_observation_file(self):
        return os.path.join(self.config.SCENE_OBSERVATION_DIR, self.config.SCENE_DATASET,
                            self.current_scene_name + '.pkl')

    @property
    def current_source_sound(self):
        # return self._source_sound_dict[self._current_sound]
        sound_sources = []
        for sound in self._current_sounds:
            sound_sources.append(self._source_sound_dict[sound])
        return sound_sources

    @property
    def is_silent(self):
        return self._episode_step_count > self._duration

    @property
    def pathfinder(self):
        return self._sim.pathfinder

    def get_agent(self, agent_id):
        return self._sim.get_agent(agent_id)

    def reconfigure(self, config: Config) -> None:
        self.config = config
        if hasattr(self.config.AGENT_0, 'OFFSET'):
            self._offset = int(self.config.AGENT_0.OFFSET)
        else:
            self._offset = 0
        if self.config.AUDIO.EVERLASTING:
            self._duration = 500
        else:
            assert hasattr(self.config.AGENT_0, 'DURATION')
            self._duration = int(self.config.AGENT_0.DURATION)
        self._audio_index = 0
        is_same_sound = config.AGENT_0.SOUND_IDs == self._current_sounds
        if not is_same_sound:
            self._current_sounds = self.config.AGENT_0.SOUND_IDs
            # self._load_single_source_sound()
            self._load_source_sounds()
            logging.debug("Switch to sound {} with duration {} seconds".format(self._current_sounds, self._duration))

        is_same_scene = config.SCENE == self._current_scene
        if not is_same_scene:
            self._current_scene = config.SCENE
            logging.debug('Current scene: {} and sound: {}'.format(self.current_scene_name, self._current_sounds))

            self._sim.close()
            del self._sim
            self.sim_config = self.create_sim_config(self._sensor_suite)
            self._sim = habitat_sim.Simulator(self.sim_config)
            self.add_acoustic_config()

            for i in range(self._audio_sensor_num):
                audio_sensor = self._sim.get_agent(0)._sensors[f"audio_sensor_{i}"]
            # audio_sensor = self._sim.get_agent(0)._sensors[f"audio_sensor"]
                audio_sensor.setAudioMaterialsJSON("data/mp3d_material_config.json")

            logging.debug('Loaded scene {}'.format(self.current_scene_name))

            self.points, self.graph = load_metadata(self.metadata_dir)

        self._update_agents_state()

        for i in range(self._audio_sensor_num):
            audio_sensor = self._sim.get_agent(0)._sensors[f"audio_sensor_{i}"]
            # audio_sensor = self._sim.get_agent(0)._sensors[f"audio_sensor"]

            # 1.5 is the offset for the height
            audio_sensor.setAudioSourceTransform(
                np.array(self.config.AGENT_0.GOAL_POSITIONs[i]) + np.array([0., 1.5, 0.])
            )

            # f = open("debug.txt", "a")
            # f.write(f"goal: {np.array(self.config.AGENT_0.GOAL_POSITIONs[i]) + np.array([0., 1.5, 0.])}\n")
            # f.write(f"audio_sensor.specification(): {audio_sensor.specification()}\n")
            # f.write(f"audio_sensor.specification().positioin: {audio_sensor.specification().position}\n")
            # f.close()

        self._episode_step_count = 0
        self._last_rir = None
        # self._current_sample_index = np.random.randint(self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME)

    @staticmethod
    def position_encoding(position):
        return '{:.2f}_{:.2f}_{:.2f}'.format(*position)

    def _position_to_index(self, position):
        if self.position_encoding(position) in self._position_to_index_mapping:
            return self._position_to_index_mapping[self.position_encoding(position)]
        else:
            raise ValueError("Position misalignment.")
    
    def update_goals(self):
        delete_goal_id, _ = self.closest_goal_id_and_dis()
        self.not_found_goals = [goal_id for goal_id in self.not_found_goals if goal_id != delete_goal_id]
        if len(self.not_found_goals) != 0:
            goals = [self.goals_dict[goal_id] for goal_id in self.not_found_goals]
            geo_dis_from_goals = self.calc_geo_dis_from_goals(goals)
            self.distance_from_goals = {goal_id: geo_dis_from_goals[i] for i, goal_id in enumerate(self.not_found_goals)}
        else:
            self.distance_from_goals = {}

    def closest_goal_id_and_dis(self):
        """
        現在地から、not foundのゴールで一番近いもののgoal_idと距離を計算する
        """
        current_postion = self.get_agent_state().position
        goals = [self.goals_dict[goal_id] for goal_id in self.not_found_goals]
        geo_dis_to_goals = self.calc_geo_dis_to_goals(current_postion, goals)
        closest_goal_id = self.not_found_goals[np.argmin(geo_dis_to_goals)]
        closest_goal_dis = np.min(geo_dis_to_goals)
        return closest_goal_id, closest_goal_dis
    
    def geodesic_distance(
        self,
        position_a: List[float],
        position_bs: List[List[float]],
        episode=None,
        ):
        """
        合計の最短のgeodesic_distanceを計算する
        """
        current_position = self.get_agent_state().position
        goals = [self.goals_dict[goal_id] for goal_id in self.not_found_goals]
        geo_dis_to_goals = self.calc_geo_dis_to_goals(current_position, goals)
        distance_to_goals = {goal_id : geo_dis_to_goals[i] for i, goal_id in enumerate(self.not_found_goals)}

        min_geo_dis = np.inf
        for goal_id in self.not_found_goals:
            d = distance_to_goals[goal_id] + self.distance_from_goals[goal_id]
            if d < min_geo_dis:
                min_geo_dis = d
            
        # f = open("debug.txt", "a")
        # f.write("---------------- geo dis --------------------\n")
        # f.write(f"current_position: {current_position}\n")
        # f.write(f"not_found_goals: {self.not_found_goals}\n")
        # f.write(f"goals_dict: {self.goals_dict}\n")
        # f.write(f"distance_from_goals: {self.distance_from_goals}\n")
        # f.write(f"distance_to_goals: {distance_to_goals}\n")
        # f.write(f"min_geo_dis: {min_geo_dis}\n")
        # f.write("----------------- fin geo dis ----------------\n")
        # f.close()
        return min_geo_dis

    def reset(self):
        # f = open("debug.txt", "a")
        # f.write("-------- reset in continuous_simulator ------\n")
        # f.close()
        # s = time.time()

        logging.debug('Reset simulation')
        sim_obs = self._sim.reset()

        # obs_s = time.time()
        if self._update_agents_state():
            sim_obs = self._sim.get_sensor_observations()
        # obs_time = time.time() - obs_s

        # observations_s = time.time()
        self._is_episode_active = True
        self._prev_sim_obs = sim_obs
        self._previous_step_collided = False
        # Encapsule data under Observations class
        observations = self._sensor_suite.get_observations(sim_obs)
        # observations_time = time.time() - observations_s

        # goal_s = time.time()
        goals = self.config.AGENT_0.GOAL_POSITIONs
        self.not_found_goals = [f"goal_{i}" for i in range(len(goals))]
        self.goals_dict = {f"goal_{i}": goal for i, goal in enumerate(goals)}
        geo_dis_from_goals = self.calc_geo_dis_from_goals(goals)
        self.distance_from_goals = {f"goal_{i}": geo_dis_from_goals[i] for i in range(len(geo_dis_from_goals))} # 各ゴールから、すべてのゴールに到達するときの最短距離
        # goal_time = time.time() - goal_s

        # f = open("debug.txt", "a")
        # f.write(f"not_found_goals: {self.not_found_goals}\n")
        # f.write(f"goals_dict: {self.goals_dict}\n")
        # f.write(f"distance_from_goals: {self.distance_from_goals}\n")
        # f.close()

        # f = open("debug.txt", "a")
        # f.write(f"obs_time: {obs_time}\n")
        # f.write(f"observations_time: {observations_time}\n")
        # f.write(f"goal_time: {goal_time}\n")
        # f.write(f"FIN reset in ContinuousSimulator: {time.time() - s}[s]\n")
        # f.close()

        return observations

    def step(self, action):
        """
        All angle calculations in this function is w.r.t habitat coordinate frame, on X-Z plane
        where +Y is upward, -Z is forward and +X is rightward.
        Angle 0 corresponds to +X, angle 90 corresponds to +y and 290 corresponds to 270.

        :param action: action to be taken
        :return:
        Dict of observations
        """
        # s = time.time()
        assert self._is_episode_active, (
            "episode is not active, environment not RESET or "
            "STOP action called previously"
        )

        self._last_rir = {}
        for i in range(self._audio_sensor_num):
            self._last_rir[f"audio_sensor_{i}"] = np.transpose(np.array(self._prev_sim_obs[f"audio_sensor_{i}"]))
            # self._last_rir[f"audio_sensor_{i}"] = np.transpose(np.array(self._prev_sim_obs[f"audio_sensor"]))

        sim_obs = self._sim.step(action)
        # {"rgb": ndarray(四次元なのでrgbd?), "audio_sensor": list, "collied": bool}

        # for i in range(self._audio_sensor_num):
        #     f = open("debug.txt", "a")
        #     f.write(f"shape audio_sensor_{i}: {np.shape(sim_obs[f'audio_sensor_{i}'])}\n")
        #     f.close()

        self._prev_sim_obs = sim_obs
        observations = self._sensor_suite.get_observations(sim_obs)
        # {"rgb": ndarray (3次元になってる)}

        self._episode_step_count += 1
        self._current_sample_index = int(self._current_sample_index + self.config.AUDIO.RIR_SAMPLING_RATE *
                                         self.config.STEP_TIME) % self.current_source_sound[0].shape[0]
        
        # f = open("debug.txt", "a")
        # f.write(f"FIN step in ContinuousSimulator: {time.time() - s}[s]\n")
        # f.close()
        return observations

    def _load_source_sounds(self):
        # load all mono files at once
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            # sound = sound_file.split('.')[0]
            sound = sound_file
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            self._source_sound_dict[sound] = audio_data
            self._audio_length = audio_data.shape[0] // self.config.AUDIO.RIR_SAMPLING_RATE

    def _load_single_source_sound(self):
        if self._current_sounds[0] not in self._source_sound_dict:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, self._current_sounds[0]),
                                          sr=self.config.AUDIO.RIR_SAMPLING_RATE)
            if audio_data.shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE == 1:
                audio_data = np.concatenate([audio_data] * 3, axis=0)  # duplicate to be longer than longest RIR
            self._source_sound_dict[self._current_sounds[0]] = audio_data
        self._audio_length = self._source_sound_dict[self._current_sounds[0]].shape[0]//self.config.AUDIO.RIR_SAMPLING_RATE

    def _compute_audiogoal(self):
        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        if self._episode_step_count > self._duration:
            logging.debug('Step count is greater than duration. Empty spectrogram.')
            audiogoal = np.zeros((2, sampling_rate))
        else:
            audiogoal = None
            for i in range(self._audio_sensor_num):
                if not f"goal_{i}" in self.not_found_goals:
                    continue

                # f = open("debug.txt", "a")
                # f.write(f"i: {i}, {list(self._source_sound_dict.keys())[i]}\n")
                # f.close()
                # i: 0, telephone.wav
                # i: 1, bell.wav
                # i: 2, birds1.wav

                binaural_rir = np.transpose(np.array(self._prev_sim_obs[f"audio_sensor_{i}"]))
                # binaural_rir = np.transpose(np.array(self._prev_sim_obs[f"audio_sensor"]))

                sound_source = np.array(list(self.current_source_sound[i]))

                # if i == 2:
                #     va = 5
                # else:
                #     va = 1
                va = 1

                if audiogoal is None:
                    audiogoal = self._convolve_with_rir(binaural_rir, sound_source, va)
                else:
                    audiogoal += self._convolve_with_rir(binaural_rir, sound_source, va)

            # f = open("debug.txt", "a")
            # f.write(f"self.config.AUDIO.CROSSFADE: {self.config.AUDIO.CROSSFADE}\n")
            # f.write(f"self._last_rir is not None: {self._last_rir is not None}\n")
            # f.close()

            if self.config.AUDIO.CROSSFADE and self._last_rir is not None:
                audiogoal_from_last_rir = None
                for i in range(self._audio_sensor_num):
                    if not f"goal_{i}" in self.not_found_goals:
                        continue

                    # list(self._source_sound_dict.values())[i]
                    # if i == 2:
                    #     va = 5
                    # else:
                    #     va = 1
                    va = 1

                    sound_source = np.array(list(self.current_source_sound[i]))

                    if audiogoal_from_last_rir is None:
                        audiogoal_from_last_rir = self._convolve_with_rir(
                            self._last_rir[f"audio_sensor_{i}"],
                            sound_source,
                            va,
                        )
                    else:
                        audiogoal_from_last_rir += self._convolve_with_rir(
                            self._last_rir[f"audio_sensor_{i}"],
                            sound_source,
                            va,
                        )

                audiogoal = crossfade(audiogoal_from_last_rir, audiogoal, sampling_rate)

        return audiogoal

    def _convolve_with_rir(self, rir, source_sound = None, volume_adjust = 1.0):
        # if source_sound is None:
        #     source_sound = self.current_source_sound

        sampling_rate = self.config.AUDIO.RIR_SAMPLING_RATE
        num_sample = int(sampling_rate * self.config.STEP_TIME)

        # f = open("debug.txt", "a")
        # f.write(f"sampling_rate: {sampling_rate}\n")
        # f.write(f"num_sample: {num_sample}\n")
        # f.write(f"rir.shape: {rir.shape}\n")

        index = self._current_sample_index
        # f.write(f"index: {index}\n")
        # f.write(f"source_sound.shape: {source_sound.shape}\n")

        if index - rir.shape[0] < 0:
            sound_segment = source_sound[: index + num_sample] * volume_adjust
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel]
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved[:, index: index + num_sample]
        else:
            # include reverb from previous time step
            if index + num_sample < source_sound.shape[0]:
                sound_segment = source_sound[index - rir.shape[0] + 1: index + num_sample]
            else:
                wraparound_sample = index + num_sample - source_sound.shape[0]
                sound_segment = np.concatenate([source_sound[index - rir.shape[0] + 1:],
                                                source_sound[: wraparound_sample]])
            # sound_segment = self.current_source_sound[index - rir.shape[0] + 1: index + num_sample]
            binaural_convolved = np.array([fftconvolve(sound_segment, rir[:, channel], mode='valid',
                                                       ) for channel in range(rir.shape[-1])])
            audiogoal = binaural_convolved

        # audiogoal = np.array([fftconvolve(self.current_source_sound, rir[:, channel], mode='full',
        #                                   ) for channel in range(rir.shape[-1])])
        # audiogoal = audiogoal[:, self._episode_step_count * num_sample: (self._episode_step_count + 1) * num_sample]

        # f.write(f"audiogoal.shape: {audiogoal.shape}\n")
        # f.close()

        audiogoal = np.pad(audiogoal, [(0, 0), (0, sampling_rate - audiogoal.shape[1])])
        return audiogoal

    def get_current_audiogoal_observation(self):
        return self._compute_audiogoal()

    def get_current_spectrogram_observation(self, audiogoal2spectrogram):
        return audiogoal2spectrogram(self.get_current_audiogoal_observation())

    # def geodesic_distance(
    #     self,
    #     position_a: List[float],
    #     position_bs: List[List[float]],
    #     episode=None,
    # ) -> float:

    #     f = open("debug.txt", "a")
    #     f.write("---------------------------------- in geodesic_distance -----------------------------------------\n")
    #     f.write(f"position_bs: {position_bs}\n")
    #     f.write(f"position_a: {position_a}\n")
    #     f.write(f"episode: {episode}\n")
    #     f.write(f"episode._shortest_path_cache: {episode._shortest_path_cache}\n")
    #     f.close()

    #     if episode is None or episode._shortest_path_cache is None:
    #         path = habitat_sim.MultiGoalShortestPath()
    #         path.requested_ends = np.array(
    #             [np.array(position_bs[0], dtype=np.float32)]
    #         )
    #     else:
    #         path = episode._shortest_path_cache

    #     path.requested_start = np.array(position_a, dtype=np.float32)

    #     self.pathfinder.find_path(path)

    #     if episode is not None:
    #         episode._shortest_path_cache = path

    #     return path.geodesic_distance

    
    def _calc_geodesic_distance(self, start: List[float], goal: List[float]) -> float:
        """
        startからgoalまでのgeodesic_distanceを計算する
        """
        path = habitat_sim.MultiGoalShortestPath()
        path.requested_ends = np.array([goal], dtype=np.float32)
        path.requested_start = np.array(start, dtype=np.float32)
        self.pathfinder.find_path(path)
        return path.geodesic_distance
    
    def _calc_geo_dis_between_goals(self, goals: List[List[float]]):
        """
        すべてのゴール間の距離を計算する。
        """
        n_goal = len(goals)
        geo_dis_between_goals = np.zeros((n_goal, n_goal), dtype=np.float32)
        for i in range(len(goals)):
            for j in range(i+1, len(goals)):
                d_ij = self._calc_geodesic_distance(goals[i], goals[j])
                geo_dis_between_goals[i][j] = d_ij
                geo_dis_between_goals[j][i] = d_ij
        return geo_dis_between_goals
    
    def calc_geo_dis_from_goals(self, goals: List[List[float]]):
        """
        すべてのゴールからスタートする、その他すべてのゴールを通る最短経路を計算する
        """
        goal_ids = [id for id in range(len(goals))]
        geo_dis_between_goals = self._calc_geo_dis_between_goals(goals)
        goal_id_permutaions = list(itertools.permutations(goal_ids, len(goal_ids)))
        geo_dis_from_goals = np.full(len(goals), np.inf)
        for p in goal_id_permutaions:
            d = 0
            for i in range(len(p[:-1])):
                d += geo_dis_between_goals[p[i]][p[i+1]]
            if d < geo_dis_from_goals[p[0]]:
                geo_dis_from_goals[p[0]] = d
        return geo_dis_from_goals
    
    def calc_geo_dis_to_goals(self, start: List[float], goals: List[List[float]]) -> List[float]:
        """
        現在位置(start)から、各goalまでのgeo_disを計算する
        """
        geo_diss = []
        for goal in goals:
            geo_dis = self._calc_geodesic_distance(start, goal)
            geo_diss.append(geo_dis)
        return geo_diss
    
    def calc_min_geodesic_distance(self, start: List[float], goals: List[List[float]]) -> float:
        """
        最小のgeodesic distanceを計算する
        """
        n_goal = len(goals)
        geo_dis_to_goals = self.calc_geo_dis_to_goals(start, goals)
        geo_dis_from_goals = self.calc_geo_dis_from_goals(goals)
        min_geo_dis = np.inf
        for i in range(n_goal):
            d = geo_dis_to_goals[i] + geo_dis_from_goals[i]
            if d < min_geo_dis:
                min_geo_dis = d
        return min_geo_dis

    @property
    def previous_step_collided(self):
        return self._previous_step_collided

    def seed(self, seed):
        self._sim.seed(seed)

    def get_observations_at(
            self,
            position: Optional[List[float]] = None,
            rotation: Optional[List[float]] = None,
            keep_agent_at_new_pose: bool = False,
            found: bool = False,
    ) -> Optional[Observations]:
        current_state = self.get_agent_state()
        if position is None or rotation is None: # ActionStopから呼ばれた場合はこっち
            success = True
        else:
            success = self.set_agent_state(
                position, rotation, reset_sensors=False
            )

        if success:

            if found:
                self._last_rir = {}
                for i in range(self._audio_sensor_num):
                    self._last_rir[f"audio_sensor_{i}"] = np.transpose(np.array(self._prev_sim_obs[f"audio_sensor_{i}"]))

            sim_obs = self._sim.get_sensor_observations()

            self._prev_sim_obs = sim_obs

            observations = self._sensor_suite.get_observations(sim_obs)
            if not keep_agent_at_new_pose: # ActionStopから呼ばれた場合
                self.set_agent_state(
                    current_state.position,
                    current_state.rotation,
                    reset_sensors=False,
                )

            if found:
                self._episode_step_count += 1
                self._current_sample_index = int(
                    self._current_sample_index + self.config.AUDIO.RIR_SAMPLING_RATE * self.config.STEP_TIME
                ) % self.current_source_sound[0].shape[0]
            return observations
        else:
            return None

    def get_straight_shortest_path_points(self, position_a, position_b):
        path = habitat_sim.ShortestPath()
        path.requested_start = position_a
        path.requested_end = position_b
        self.pathfinder.find_path(path)
        return path.points

    def make_greedy_follower(self, *args, **kwargs):
        return self._sim.make_greedy_follower(*args, **kwargs)
