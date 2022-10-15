from logging.handlers import RotatingFileHandler
import time
from typing import List, Any, Dict
import itertools
import gzip
import json
import random as rand
import argparse
import os
import sys
import pprint

from habitat_sim import PathFinder, ShortestPath, Simulator, NavMeshSettings, SimulatorConfiguration
from habitat_sim.bindings import Random
from habitat_sim.utils.common import quat_from_angle_axis
from habitat.sims.habitat_simulator.habitat_simulator import overwrite_config

sys.path.insert(0, "../habitat-sim")
# print(sys.path)
import examples.settings

import numpy as np
import yaml


# def get_sim(scene: str, seed: int, move_forward_amount: float, turn_left_amount: int, turn_right_amount: int) -> "Simulator":
#     f = open("debug.txt", "a")
#     f.write("-- pathfinder default nav_mesh_settings --\n")
    # pathfinder = PathFinder()
    # pathfinder.seed(seed)
    # navmesh_filename = f"data/scene_datasets/replica/{scene}/habitat/mesh_semantic.navmesh"
    # pathfinder.load_nav_mesh(navmesh_filename)
    # navmesh_settings = pathfinder.nav_mesh_settings


    # cfg_settings = examples.settings.default_sim_settings.copy()
    # cfg_settings["physics_config_file"] = "/home/haru/av-nav/habitat-sim/data/default.physics_config.json"
    ## cfg_settings["scene"] = f"./data/scene_datasets/replica/{scene}"
    # hab_cfg = examples.settings.make_cfg(cfg_settings)
    # hab_cfg.sim_cfg.scene_id = f"./data/scene_datasets/replica/{scene}/habitat/mesh_semantic.ply"

    # f.write("-- cfg_settings --------------------\n")
    # pprint.pprint(cfg_settings, f)

    # hab_cfg = SimulatorConfiguration()
    # hab_cfg.scene_id = f"./data/scene_datasets/replica/{scene}/habitat/mesh_semantic.ply"
    # f.write(f"hab_cfg.enable_physics: {hab_cfg.enable_physics}\n")
    # hab_cfg.enable_physics = False

    # sim = Simulator(hab_cfg)
    # sim.reconfigure(hab_cfg)
    
    # navmesh_settings = NavMeshSettings()
    # navmesh_settings.set_defaults()

    # assert sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    # assert sim.pathfinder.is_loaded
    # assert len(sim.pathfinder.build_navmesh_vertices()) > 0
    # assert len(sim.pathfinder.build_navmesh_vertex_indices()) > 0

    # f.write(f"sim.pathfinder.navigatable_area\n{sim.pathfinder.navigable_area}\n")
    # f.close()
    # return sim

def get_pathfinder(scene: str, seed: int):
    pathfinder = PathFinder()
    pathfinder.seed(seed)
    navmesh_filename = f"data/scene_datasets/replica/{scene}/habitat/mesh_semantic.navmesh"
    pathfinder.load_nav_mesh(navmesh_filename)
    return pathfinder

def get_random(seed: int) -> "Random":
    random = Random()
    random.seed(seed)
    return random

def calc_euclid_distance(p_a: List[float], p_b: List[float]) -> float:
    p_a = np.array(p_a)
    p_b = np.array(p_b)
    euclid_dis = np.linalg.norm(p_a - p_b)
    return float(euclid_dis)

def calc_geodesic_distance(pathfinder: "Pathfinder", start: List[float], goal: List[float]) -> float:
    """
    startからgoalまでのgeodesic_distanceを計算する
    """
    path = ShortestPath()
    path.requested_end = np.array(goal, dtype=np.float32)
    path.requested_start = np.array(start, dtype=np.float32)
    pathfinder.find_path(path)
    return path.geodesic_distance


def calc_geo_dis_between_goals(pathfinder: "Pathfinder", goals: List[List[float]]) -> List[List[float]]:
    """
    すべてのゴール間の距離を計算する。
    """
    n_goal = len(goals)
    geo_dis_between_goals = np.zeros((n_goal, n_goal), dtype=np.float32)
    for i in range(len(goals)):
        for j in range(i+1, len(goals)):
            d_ij = calc_geodesic_distance(pathfinder, goals[i], goals[j])
            geo_dis_between_goals[i][j] = d_ij
            geo_dis_between_goals[j][i] = d_ij
    return list(geo_dis_between_goals)


def calc_geo_dis_from_goals(pathfinder: "Pathfinder", goals: List[List[float]]) -> List[float]:
    """
    すべてのゴールからスタートする、その他すべてのゴールを通る最短経路を計算する
    """
    goal_ids = [id for id in range(len(goals))]
    geo_dis_between_goals = calc_geo_dis_between_goals(pathfinder, goals)
    goal_id_permutaions = list(itertools.permutations(goal_ids, len(goal_ids)))
    geo_dis_from_goals = np.full(len(goals), np.inf)
    for p in goal_id_permutaions:
        d = 0
        for i in range(len(p[:-1])):
            d += geo_dis_between_goals[p[i]][p[i+1]]
        if d < geo_dis_from_goals[p[0]]:
            geo_dis_from_goals[p[0]] = d
    return list(geo_dis_from_goals)


def calc_geo_dis_to_goals(pathfinder: "Pathfinder", start: List[float], goals: List[List[float]]) -> List[float]:
    """
    現在位置(start)から、各goalまでのgeo_disを計算する
    """
    geo_diss = []
    for goal in goals:
        geo_dis = calc_geodesic_distance(pathfinder, start, goal)
        geo_diss.append(geo_dis)
    return geo_diss


def get_random_position(pathfinder: "Pathfinder") -> List[float]:
    while True:
        position = pathfinder.get_random_navigable_point(max_tries=10)
        if not np.isnan(position[0]):
            break
    return list(np.array(position, dtype=float))


def get_random_rotation(random: "Random") -> List[float]:
    rotation_quat = quat_from_angle_axis(
        random.uniform_float(0, 2.0 * np.pi), np.array([0, 1, 0])
    )
    rotation = [rotation_quat.x, rotation_quat.y, rotation_quat.z, rotation_quat.w]
    return rotation


def get_random_goal_position(pathfinder: "Pathfinder", start: List[float], goals: List[List[float]], step_size: float) -> List[float]:
    """
    goalsはすでに決定されているgoalたち。ゴール間距離とかを図るために使用
    """
    exist_path = False
    easy = True
    almost_linear = True
    same_height = False # startとgoalが同じ高さであるかどうか
    other_points = [start] + goals
    # other_points = [start]
    cnt = 0
    while not (exist_path and not easy and not almost_linear and same_height):
        if cnt > 100:
            return None
        cnt += 1
        goal = get_random_position(pathfinder)
        geo_dis = calc_geodesic_distance(pathfinder, start, goal)

        exist_path = True
        almost_linear = False
        easy = False
        same_height = True

        if geo_dis == np.inf:
            exist_path = False
            continue

        for p in other_points:
            geo_d = calc_geodesic_distance(pathfinder, goal, p)
            euclid_d = calc_euclid_distance(goal, p)
            if geo_d <= 4*step_size:
                easy = True
                break
            if geo_d / euclid_d <= 1.1:
                almost_linear = True
                break
            if goal[1] != p[1]:
                same_height = False
                break
    return goal


def calc_min_geodesic_distance(pathfinder: "Pathfinder", start: List[float], goals: List[List[float]]) -> float:
    n_goal = len(goals)
    geo_dis_to_goals = calc_geo_dis_to_goals(pathfinder, start, goals)
    geo_dis_from_goals = calc_geo_dis_from_goals(pathfinder, goals)
    min_geo_dis = np.inf
    for i in range(n_goal):
        d = geo_dis_to_goals[i] + geo_dis_from_goals[i]
        if d < min_geo_dis:
            min_geo_dis = d
    return min_geo_dis


def get_random_sounds(n_sound: int, sounds: List[str]) -> List[str]:
    random_sounds = rand.sample(sounds, n_sound)
    return random_sounds


def calc_num_action() -> int:
    # これは行動空間にいぞんしているよね？
    # TODO とりあえず使わない予定だけど未実装なので、必要になったら
    return None

def get_start_and_goals(pathfinder, random, sounds, step_size):
    n_goal = len(sounds)
    start_position = get_random_position(pathfinder)
    start_rotation = get_random_rotation(random)
    goals = []
    for _ in range(n_goal):
        goal_position = get_random_goal_position(
            pathfinder,
            start_position,
            [list(goal["position"]) for goal in goals],
            step_size,
        )
        if goal_position is None:
            goals = None
            break
        goal_radius = 1e-5
        goal = {
            "position": goal_position,
            "radius": goal_radius,
        }
        goals.append(goal)
    return start_position, start_rotation, goals
    

def make_episode(
    pathfinder: "Pathfinder",
    random: "Random",
    episode_id: int,
    scene_name: str,
    sounds: List[str],
    n_sound: int,
    step_size: float,
) -> Dict[str, Any]:
    """
    episodeのdictを作成する
    """
    # f = open("debug.txt", "a")
    # f.write("----------------- make_episode --------------\n")
    # f.close()

    while True:
        start_position, start_rotation, goals = get_start_and_goals(
            pathfinder, random, sounds, step_size
        )
        if goals is not None:
            break
    
    geodesic_distance = calc_min_geodesic_distance(
        pathfinder,
        start_position,
        [list(goal["position"]) for goal in goals],
    )
    num_action = calc_num_action()
    # sounds = get_random_sounds(n_sound, sounds)
    
    episode = {
        "episode_id": str(episode_id),
        "scene_id": f"{scene_name}/habitat/mesh_semantic.ply",
        "start_position": start_position,
        "start_rotation": start_rotation,
        "info": {
            "geodesic_distance": float(geodesic_distance),
            "num_action": num_action,
            "sounds": sounds
        },
        "goals": goals,
    }
    # f = open("debug.txt", "a")
    # f.write(f"start_position: {start_position}, start_rotation: {start_rotation}\n")
    # f.write(f"goal positions:\n")
    # for goal in goals:
    #     f.write(f"{list(goal['position'])}\n")
    # f.close()

    return episode


def make_dataset(
    n_episode: int,
    sounds: List[str],
    n_sound: int,
    scene: str,
    seed: int,
    step_size: float,
    start_episode_id: int,
) -> Dict[str, Any]:

    random = get_random(seed)
    pathfinder = get_pathfinder(scene, seed)
    episodes = []
    cnt = 0

    print(f"Making {scene} episodes")
    for episode_id in range(start_episode_id, start_episode_id + n_episode):
        cnt += 1
        print(f"\r{cnt}/{n_episode}", end="")
        episode = make_episode(
            pathfinder,
            random,
            episode_id,
            scene,
            sounds,
            n_sound,
            step_size,
        )
        episodes.append(episode)
    print()

    dataset = {
        "episodes": episodes,
        "scene": scene,
    }
    return dataset


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write("make_dataset !\n")
    f.close()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        help="path to config file",
        type=str,
        default="./configs/audionav/av_nav/replica/make_dataset.yaml"
    )
    args = parser.parse_args()
    with open(args.yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    rand.seed(config["episode"]["seed"])

    start_time = time.time()

    dataset = make_dataset(
        n_episode=config["episode"]["n_episode"],
        sounds=config["episode"]["sounds"],
        n_sound=config["episode"]["n_sound"],
        scene=config["episode"]["scene"],
        seed=config["episode"]["seed"],
        step_size=config["episode"]["step_size"],
        start_episode_id=config["episode"]["start_episode_id"]
    )
    dataset_json_str = json.dumps(dataset)

    os.makedirs(config["episode"]["save_dir_path"], exist_ok=True)

    if config["episode"]["is_train"]:
        os.makedirs(f"{config['episode']['save_dir_path']}/content", exist_ok=True)
        with gzip.open(f"{config['episode']['save_dir_path']}/content/{config['episode']['scene']}.json.gz", mode="wt") as f:
            f.write(dataset_json_str)
        
        file_name = config['episode']['save_dir_path'].split("/")[-1]
        with gzip.open(f"{config['episode']['save_dir_path']}/{file_name}.json.gz", mode="wt") as f:
            json_str = json.dumps({'episodes': []})
            f.write(json_str)
    else:
        with gzip.open(f"{config['episode']['save_dir_path']}/{config['episode']['scene']}.json.gz", mode="wt") as f:
            f.write(dataset_json_str)
        
    print(f"TOTAL TIME: {(time.time() - start_time)/60} [min]")
    