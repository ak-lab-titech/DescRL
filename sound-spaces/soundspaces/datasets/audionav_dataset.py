# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import gzip
import json
import os
import logging
import pprint
from typing import List, Optional

import numpy as np
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    ShortestPathPoint,
)

np.random.seed(0)

ALL_SCENES_MASK = "*"
CONTENT_SCENES_PATH_FIELD = "content_scenes_path"
DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


@registry.register_dataset(name="AudioNav")
class AudioNavDataset(Dataset):
    r"""Class inherited from Dataset that loads Audio Navigation dataset.
    """

    episodes: List[NavigationEpisode]
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def get_scenes_to_load(config: Config) -> List[str]:
        r"""Return list of scene ids for which dataset has separate files with
        episodes.
        """
        assert AudioNavDataset.check_config_paths_exist(config), \
            (config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT), config.SCENES_DIR)
        dataset_dir = os.path.dirname(
            config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        )

        cfg = config.clone()
        cfg.defrost()
        cfg.CONTENT_SCENES = []
        dataset = AudioNavDataset(cfg)
        return AudioNavDataset._get_scenes_from_folder(
            content_scenes_path=dataset.content_scenes_path,
            dataset_dir=dataset_dir,
        )

    @staticmethod
    def _get_scenes_from_folder(content_scenes_path, dataset_dir):
        scenes = []
        content_dir = content_scenes_path.split("{scene}")[0]
        scene_dataset_ext = content_scenes_path.split("{scene}")[1]
        content_dir = content_dir.format(data_path=dataset_dir)
        if not os.path.exists(content_dir):
            return scenes

        for filename in os.listdir(content_dir):
            if filename.endswith(scene_dataset_ext):
                scene = filename[: -len(scene_dataset_ext)]
                scenes.append(scene)
        scenes.sort()
        return scenes

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        datasetfile_path = config.DATA_PATH.format(version=config.VERSION, split=config.SPLIT)
        # f = open("debug.txt", "a")
        # f.write("--------------------- __init__ in AudioNavDataset --------------------\n\n")
        # f.write(f"datasetfile_path: {datasetfile_path}\n")
        # f.close()
        with gzip.open(datasetfile_path, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=datasetfile_path)

        # Read separate file for each scene
        dataset_dir = os.path.dirname(datasetfile_path)
        scenes = config.CONTENT_SCENES
        if ALL_SCENES_MASK in scenes:
            scenes = AudioNavDataset._get_scenes_from_folder(
                content_scenes_path=self.content_scenes_path,
                dataset_dir=dataset_dir,
            )
        
        # f = open("debug.txt", "a")
        # f.write(f"scenes: {scenes}\n")
        # f.close()

        last_episode_cnt = 0
        for scene in scenes:
            scene_filename = self.content_scenes_path.format(
                data_path=dataset_dir, scene=scene
            )
            # f = open("debug.txt", "a")
            # f.write(f"scene_filename: {scene_filename}\n")
            # f.close()
            with gzip.open(scene_filename, "rt") as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR, scene_filename=scene_filename)

            num_episode = len(self.episodes) - last_episode_cnt
            last_episode_cnt = len(self.episodes)
            logging.info('Sampled {} from {}'.format(num_episode, scene))

    def filter_by_ids(self, scene_ids):
        episodes_to_keep = list()

        for episode in self.episodes:
            for scene_id in scene_ids:
                scene, ep_id = scene_id.split(',')
                if scene in episode.scene_id and ep_id == episode.episode_id:
                    episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    # filter by scenes for data collection
    def filter_by_scenes(self, scene):
        episodes_to_keep = list()

        for episode in self.episodes:
            episode_scene = episode.scene_id.split("/")[3]
            if scene == episode_scene:
                episodes_to_keep.append(episode)

        self.episodes = episodes_to_keep

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None, scene_filename: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        # f = open("debug.txt", "a")
        # f.write("\n---------------- from_json in AudioNavDataset -------------------------\n")
        # # f.write(f"json_str: {json_str}\n\n")
        # f.write(f"scenes_dir: {scenes_dir}\n\n")
        # f.write(f"deserialized:\n")
        # # pprint.pprint(deserialized, f)
        # for k, v in deserialized.items():
        #     f.write(f"k={k}, len(v)={len(v)}, type(v)={type(v)}\n")
        # f.close()

        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        episode_cnt = 0
        f = open(f"debug.txt", "a")
        f.write("--------------------------------------------------\n")
        f.write(f"SAME_TYPE: {self._config.SOUND_TYPE}\n")
        for episode in deserialized["episodes"]:
            episode = NavigationEpisode(**episode)

            if episode_cnt == 0:
                f.write(f"episode.info['sounds'] Before change: {episode.info['sounds']}\n")
            
            n_sound = len(episode.info['sounds'])

            if self._config.SOUND_TYPE == "default":
                pass
            elif self._config.SOUND_TYPE == "all_telephone":
                episode.info['sounds'] = ['telephone' for _ in range(n_sound)]
            elif self._config.SOUND_TYPE == "all_bell":
                episode.info['sounds'] = ['bell' for _ in range(n_sound)]
            elif self._config.SOUND_TYPE == "long_and_short" or self._config.SOUND_TYPE == "long" or self._config.SOUND_TYPE == "short":
                longs = [
                    "creak", "engine_4", "leak", "propeller", "turbine_4", "fan_6", "waves4", "helicopter", "water_waves_2",
                ]
                shorts = [
                    "beeps", "birds6", "birds5", "horn_2", "infinitely", "canon_short_2", "telephone"
                ]
                # 最大振幅平均
                # LONG: 0.5851542154947916
                # SHORT: 0.5965837751116071
                if self._config.SOUND_TYPE == "long_and_short":
                    episode.info['sounds'] = [np.random.choice(longs, 1)[0], np.random.choice(shorts, 1)[0]]
                elif self._config.SOUND_TYPE == "long":
                    episode.info['sounds'] = list(np.random.choice(longs, n_sound))
                else:
                    episode.info['sounds'] = list(np.random.choice(shorts, n_sound))
            elif self._config.SOUND_TYPE == "big_and_small" or self._config.SOUND_TYPE == "big" or self._config.SOUND_TYPE == "small":
                bigs = [
                    "fan", "telephone"
                ]
                smalls = [
                    "leak", "canon_short_2"
                ]
                # fan:           0.839263916015625
                # telephone:     0.90802001953125
                # AVE:           0.8736419677734375
                # leak:          0.244171142578125
                # canon_short_2: 0.363983154296875
                # AVE:           0.3040771484375
                if self._config.SOUND_TYPE == "big_and_small":
                    episode.info['sounds'] = [np.random.choice(bigs, 1)[0], np.random.choice(smalls, 1)[0]]
                elif self._config.SOUND_TYPE == "big":
                    episode.info['sounds'] = list(np.random.choice(bigs, n_sound))
                else:
                    episode.info['sounds'] = list(np.random.choice(smalls, n_sound))
            else:
                if self._config.SOUND_TYPE == "multi_train":
                    # num: 73
                    all_sounds = [
                        'copy_machine', 'machinery', 'big_engine_loop', 'whirr_loop', 'computer_fan_2', 'fan_2', 'elevator', 'impulse', 'fan_1', 'turbine_3', 'ship_ambience', 'come_to_office', 'police_siren', 'waves3', 'angel', 'turbine_2', 'horn_beeps', 'big_door', 'electric_buzz', 'evolves', 'computer_beeps', 'lawrence', 'car_start_run', 'static', 'bell', 'waves2', 'alien_ambience', 'air_compressor_clicks', 'birds3', 'person_6', 'machinery_loop', 'wooddoorclose', 'person_0', 'metaldoorclose', 'bowl', 'air_pump', 'canon_short', 'apollo', 'person_4', 'horn_1', 'person_1', 'beep_noise_loop', 'computer_fan_1', 'plane_old', 'alarm', 'big_bass_rattle', 'birds2', 'person_3', 'air_conditioner', 'test', 'waves1', 'force_field_loop', 'chiller', 'exhaust_fan', 'steam_door', 'person_2', 'engine_2', 'person_5', 'siren', 'birds4', 'plane', 'engine_1', 'fireplace', 'water_waves_1', 'metaldooropen', 'turbine_1', 'fountain', 'fan_3', 'elevator_door_close', 'engine_3', 'sweet_beat', 'birds1', 'helicopter2'
                    ]
                elif self._config.SOUND_TYPE == "multi_val":
                    # val (num: 11)
                    all_sounds = [
                        'fan_6', 'reverb_time', 'fan_4', 'terminal', 'water_waves_2', 'come_again', 'infinitely', 'person_8', 'birds5', 'person_7', 'helicopter'
                    ]
                elif self._config.SOUND_TYPE == "multi_test":
                    # test (num: 18)
                    all_sounds = [
                        'fan_7', 'arrived', 'waves4', 'canon_short_2', 'radio_static', 'person_9', 'person_11', 'creak', 'birds6', 'turbine_4', 'person_10', 'engine_4', 'telephone', 'beeps', 'propeller', 'leak', 'horn_2', 'fan'
                    ]
                else:
                    raise NotImplementedError()

                # random
                episode.info['sounds'] = list(np.random.choice(all_sounds, n_sound))

                # same
                # sound = list(np.random.choice(all_sounds, 1))[0]
                # episode.info['sounds'] = [sound for _ in range(n_sound)]

                # different
                # episode.info['sounds'] = list(np.random.choice(all_sounds, n_sound, replace=False))


            if episode_cnt == 0:
                f.write(f"episode.info['sounds'] After change: {episode.info['sounds']}\n")

            # print(f"Sounds: {episode.info['sounds']}")

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX):
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            for g_index, goal in enumerate(episode.goals):
                episode.goals[g_index] = NavigationGoal(**goal)
            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for p_index, point in enumerate(path):
                        path[p_index] = ShortestPathPoint(**point)

            if hasattr(self._config, 'CONTINUOUS') and self._config.CONTINUOUS:
                # TODO: fix
                for i in range(len(episode.goals)):
                    episode.goals[i].position[1] += 0.1

            self.episodes.append(episode)
            episode_cnt += 1
        
        f.write(f"episode_cnt: {episode_cnt}\n")
        f.close()
