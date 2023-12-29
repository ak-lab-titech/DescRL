import time
import os
import sys
import argparse

import librosa
import numpy as np
from skimage.measure import block_reduce
import lmdb
import msgpack_numpy

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")
sys.path.append("/home/0/19B30511/av-nav/myss")

from habitat.datasets import make_dataset
from habitat.sims import make_sim
from ss_baselines.savi.config.default import get_config
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import (
    compute_spectrogram,
    compute_pose,
    get_category,
    compute_pointgoal_with_gps_compass
)


np.random.seed(0)


def get_obs_seq(num_step, sim, episode):
    image_seq_list = []
    audio_seq_list = []
    pose_seq_list = []
    action_seq_list = []
    oracle_actions = sim.get_oracle_actions_from_current_pos()

    for i in range(num_step):
        sim_obs = sim._get_sim_observation()
        observations = sim._sensor_suite.get_observations(sim_obs)
        image_shape = np.shape(observations["depth"])
    
        if len(image_shape) == 4:
            depth_img = np.squeeze(observations["depth"], axis=3)
        else:
            depth_img = observations["depth"]
        rgb_img = observations["rgb"] / 255.0
        
        if "semantic" in observations.keys():
            semantic_img = np.squeeze(observations["semantic"], axis=3) / 255.0
            image = np.concatenate([rgb_img, depth_img, semantic_img], 2).astype(np.float32)
        else:
            image = np.concatenate([rgb_img, depth_img], 2).astype(np.float32)
        
        audio = sim.get_current_spectrogram_observation(compute_spectrogram)
        pose = compute_pose(episode, sim.get_agent_state(), i)
        action = oracle_actions[i]

        image_seq_list.append([image])
        audio_seq_list.append([audio])
        pose_seq_list.append([pose])
        action_seq_list.append([action])

        sim.step(action)

    image_seq = np.array(image_seq_list)
    audio_seq = np.array(audio_seq_list)
    pose_seq = np.array(pose_seq_list)
    action_seq = np.array(action_seq_list)
    
    return image_seq, audio_seq, pose_seq, action_seq


def make_episode_data(sim_cfg, sim, episode):
    sim_cfg = merge_sim_episode_config(sim_cfg, episode)
    sim.reconfigure(sim_cfg)
    _ = sim.reset()
    
    num_step = np.random.randint(1, episode.info["num_action"])
    category = get_category(episode)
    location = compute_pointgoal_with_gps_compass(sim.get_agent_state(), episode)
    image_seq, audio_seq, pose_seq, action_seq = get_obs_seq(num_step, sim, episode)
    instruction = np.array(episode.instructions)[:, num_step-1]
    data = [
        image_seq,
        audio_seq,
        pose_seq,
        action_seq,
        category,
        location,
        instruction,
    ]
    return data


def main(config, sensor_list, save_path):
    dataset = make_dataset(
        id_dataset=config.DATASET.TYPE,
        config=config.DATASET,
    )
    episodes = dataset.episodes

    sim_cfg = config.SIMULATOR
    sim_cfg.defrost()
    sim_cfg.SCENE_DATASET = episodes[0].scene_dataset_config
    sim_cfg.AGENT_0.SENSORS = sensor_list
    sim_cfg.SCENE = episodes[0].scene_id
    sim_cfg.freeze()
    sim = make_sim(
        sim_cfg.TYPE, config=sim_cfg,
    )
    s = time.time()
    for i, episode in enumerate(episodes):
        data = make_episode_data(sim_cfg, sim, episode)

        env = lmdb.open(save_path, map_size=int(1.2e12))
        with env.begin(write=True) as txn:
            txn.put(
                f"{i}".encode(), 
                msgpack_numpy.packb(
                    data, use_bin_type=True
                ),
            )
        if i % 1000 == 0:
            f = open("debug.txt", "a")
            f.write(f"i: {i}/{len(episodes)}\n")
            f.write(f"TIME: {time.time() - s}\n")
            f.close()
            s = time.time()


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START make_episode_dataset!\n")
    f.close()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--save-dataset-path', type=str)
    args = parser.parse_args()

    config = get_config(args.config)
    os.makedirs(f"{args.save_dataset_path}", exist_ok=True)
    main(config.TASK_CONFIG, config.SENSORS, args.save_dataset_path)

    