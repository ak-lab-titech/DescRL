#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type
import logging
import math
import time

import habitat
from habitat import Config, Dataset
from ss_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        # import pprint
        # f = open("debug.txt", mode="a")
        # f.write("config in __init__ of AudioNavRLEnv\n")
        # pprint.pprint(config, f)
        # f.close()
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._continuous = config.CONTINUOUS

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

        # self.episode_start_time = time.time()

    def reset(self):
        # f = open("debug.txt", mode="a")
        # f.write("==================================================== RESET AudioNavRLEnv ===================================================================\n")
        # f.close()

        # f = open("debug.txt", "a")
        # f.write(f"Reset in AudioNavRL: {time.time() - self.episode_start_time} [s]\n")
        # f.close()
        # self.episode_start_time = time.time()
        
        # s = time.time()

        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)

        if self._continuous:
            self._previous_target_distance = self._distance_target()
        else:
            self._previous_target_distance = self.habitat_env.current_episode.info[
                "geodesic_distance"
            ]
        
        # f = open("debug.txt", "a")
        # f.write(f"FIN reset: {time.time() - s}[s]\n")
        # f.close()
        return observations

    def step(self, *args, **kwargs):
        # f = open("debug.txt", "a")
        # f.write("============================================== STEP AudioNavRL =======================================================\n")
        # f.write(f"Action: {kwargs['action']}\n")
        # f.close()
        # s = time.time()
        self._previous_action = kwargs["action"]
        step_return =  super().step(*args, **kwargs)
        # f = open("debug.txt", "a")
        # f.write(f"FIN step in AudioNavRL: {time.time() - s}[s]\n")
        # f.close()
        return step_return

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        """
        報酬の計算を行う。

        reward = slack_reward + (previous_distance - current_distance) * scale + success_reward
        """
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance
        

        if self._found():
            # f = open("debug.txt", "a")
            # f.write("success FOUND\n")
            # f.close()

            # reward += 1 * self._rl_config.FOUND_REWARD
            reward += 1 * 10
            self._env.sim.update_goals()
            logging.debug('Found goal!')

        if self._episode_success():
            # f = open("debug.txt", "a")
            # f.write("success SUCCESS\n")
            # f.close()
            reward += self._env.get_metrics()['success'] * self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        assert not math.isnan(reward)

        return reward

    def _distance_target(self):
        return self._env.get_metrics()['distance_to_goal']

    def _episode_success(self):
        _, distance_to_closest_target = self._env.sim.closest_goal_id_and_dis()
        if (
            self._env.task.is_found_called
            and (
                (self._continuous and distance_to_closest_target < self._success_distance) # 連続用
                or (not self._continuous and self._env.sim.reaching_goal) # 離散用
            )
            and len(self._env.sim.not_found_goals) == 1
        ):
            return True
        return False
    
    def _found(self):
        """
        FoundActionが呼ばれて成功したかどうか
        """
        _, distance_to_closest_target = self._env.sim.closest_goal_id_and_dis()
        if (
            self._env.task.is_found_called
            and self._continuous and distance_to_closest_target < self._success_distance
            and len(self._env.sim.not_found_goals) > 1
        ):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id

    def get_sound_diss(self):
        sounds_dict = {}
        for i, sound in enumerate(self._env.sim._current_sounds):
            if f"goal_{i}" in self._env.sim.not_found_goals:

                current_position = self._env.sim.get_agent_state().position
                goals = [self._env.sim.goals_dict[f"goal_{i}"]]
                geo_dis = self._env.sim.calc_geo_dis_to_goals(current_position, goals)[0]
                
            else:
                geo_dis = None
            
            if not sound in sounds_dict.keys():
                sounds_dict[sound] = [geo_dis]
            else:
                sounds_dict[sound].append(geo_dis)
                
        return sounds_dict
