#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import json

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")

os.environ['MAGNUM_LOG'] = "quiet"
os.environ['HABITAT_SIM_LOG'] = "quiet"

import argparse
from math import pi
import logging

import numpy as np

import habitat
from habitat.sims.habitat_simulator.actions import HabitatSimActions
# from habitat.config.default import get_config
from ss_baselines.common.benchmark import Benchmark
from ss_baselines.av_nav.config.default import get_task_config
# from ss_baselines.av_wan.config.default import get_task_config
from ss_baselines.common.utils import NpEncoder


class RandomAgent(habitat.Agent):
    def __init__(self, success_distance, goal_sensor_uuid, goal_num):
        self.dist_threshold_to_stop = success_distance
        self.goal_sensor_uuid = goal_sensor_uuid
        self.goal_num = goal_num
        self.found_goal_id = []

    def reset(self):
        self.found_goal_id = []

    def is_goal_reached(self, observations):
        # because the frame is in with polar coordinates

        # dist = observations[self.goal_sensor_uuid][0]
        # return dist <= self.dist_threshold_to_stop

        dists = [
            observations[self.goal_sensor_uuid][2*i] for i in range(
                int(len(observations[self.goal_sensor_uuid])/2)
            )
        ]

        for i, dist in enumerate(dists):
            if not (i in self.found_goal_id) and dist <= self.dist_threshold_to_stop:
                self.found_goal_id.append(i)
                return True
        return False

    def act(self, observations):

        reached = self.is_goal_reached(observations)
        if reached:
            action = HabitatSimActions.FOUND
        else:
            action = np.random.choice(
                [
                    HabitatSimActions.MOVE_FORWARD,
                    HabitatSimActions.TURN_LEFT,
                    HabitatSimActions.TURN_RIGHT,
                ]
            )
        # f = open("debug.txt", "a")
        # f.write(f"--------------- act ----------------\n")
        # f.write(f"NUM: {self.goal_num}, found_goal_id: {self.found_goal_id}, reached: {reached}, action: {action}\n")
        # f.close()
        return {"action": action}


class ForwardOnlyAgent(RandomAgent):
    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            action = HabitatSimActions.MOVE_FORWARD
        return {"action": action}


class RandomForwardAgent(RandomAgent):
    def __init__(self, success_distance, goal_sensor_uuid):
        super().__init__(success_distance, goal_sensor_uuid)
        self.FORWARD_PROBABILITY = 0.8

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            if np.random.uniform(0, 1, 1) < self.FORWARD_PROBABILITY:
                action = HabitatSimActions.MOVE_FORWARD
            else:
                action = np.random.choice(
                    [HabitatSimActions.TURN_LEFT, HabitatSimActions.TURN_RIGHT]
                )

        return {"action": action}


class GoalFollower(RandomAgent):
    def __init__(self, success_distance, goal_sensor_uuid):
        super().__init__(success_distance, goal_sensor_uuid)
        self.pos_th = self.dist_threshold_to_stop
        self.angle_th = float(np.deg2rad(15))
        self.random_prob = 0

    def normalize_angle(self, angle):
        if angle < -pi:
            angle = 2.0 * pi + angle
        if angle > pi:
            angle = -2.0 * pi + angle
        return angle

    def turn_towards_goal(self, angle_to_goal):
        if angle_to_goal > pi or (
            (angle_to_goal < 0) and (angle_to_goal > -pi)
        ):
            action = HabitatSimActions.TURN_RIGHT
        else:
            action = HabitatSimActions.TURN_LEFT
        return action

    def act(self, observations):
        if self.is_goal_reached(observations):
            action = HabitatSimActions.STOP
        else:
            angle_to_goal = self.normalize_angle(
                np.array(observations[self.goal_sensor_uuid][1])
            )
            if abs(angle_to_goal) < self.angle_th:
                action = HabitatSimActions.MOVE_FORWARD
            else:
                action = self.turn_towards_goal(angle_to_goal)

        return {"action": action}


def get_all_subclasses(cls):
    return set(cls.__subclasses__()).union(
        [s for c in cls.__subclasses__() for s in get_all_subclasses(c)]
    )


def get_agent_cls(agent_class_name):
    sub_classes = [
        sub_class
        for sub_class in get_all_subclasses(habitat.Agent)
        if sub_class.__name__ == agent_class_name
    ]
    return sub_classes[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--success-distance", type=float, default=0.2)
    parser.add_argument(
        "--task-config", type=str, default="configs/tasks/pointnav.yaml"
    )
    parser.add_argument("--agent-class", type=str, default="RandomAgent")
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    task_config = get_task_config(args.task_config, args.opts)

    f = open("debug.txt", "a")
    f.write(f"split: {task_config.DATASET.SPLIT}\n")
    f.write(f"version: {task_config.DATASET.VERSION}\n")
    f.close()

    agent = get_agent_cls(args.agent_class)(
        success_distance=args.success_distance,
        goal_sensor_uuid=task_config.TASK.GOAL_SENSOR_UUID,
        goal_num=task_config.SIMULATOR.AUDIO.NUM,
    )
    benchmark = Benchmark(task_config)

    metrics, all_metrics = benchmark.evaluate(agent)

    for k, v in metrics.items():
        habitat.logger.info("{}: {:.3f}".format(k, v))
        habitat.logger.info("    mean: {:.3f}, sgd: {:.3f}".format(
            np.mean(all_metrics[k]),
            np.std(all_metrics[k]),
        ))
    stats_file = os.path.join(
        f"./data/models/ss2/replica/{task_config.SIMULATOR.AUDIO.NUM}g-random",
        "{}_{}.json".format(
            task_config.DATASET.SPLIT,
            os.getenv('JOB_ID')
        )
    )
    with open(stats_file, 'w') as fo:
        json.dump(all_metrics, fo, cls=NpEncoder)


if __name__ == "__main__":
    f = open("debug.txt", "w")
    f.write("start simple_agents.py!\n")
    f.close()
    main()


# デフォルトのnum_episodes (1000)だと、普通のav-navで、Randomだと1日10時間くらいかかる。

# 50の結果
# 2022-10-17 22:19:10, INFO: Average reward: -1.9367682201521168 in 49 episodes
# 2022-10-17 22:19:10, INFO: Average episode steps: 425.1020408163265
# 2022-10-17 22:19:10, INFO: Success rate: 0.2
# 2022-10-17 22:19:10,134 distance_to_goal: 5.668
# 2022-10-17 22:19:10,134 normalized_distance_to_goal: 1.097
# 2022-10-17 22:19:10,134 success: 0.204
# 2022-10-17 22:19:10,135 spl: 0.135
# 2022-10-17 22:19:10,135 softspl: 0.174
# 2022-10-17 22:19:10,135 na: 425.102
# 2022-10-17 22:19:10,135 sna: 0.044
