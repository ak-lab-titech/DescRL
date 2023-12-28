import sys
import os
import time

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.measure import block_reduce

from habitat.datasets import make_dataset
from habitat.sims import make_sim
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config
from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING


# TODO seedの設定

class IPRLPretrainingDataset(Dataset):
    def __init__(self, config, sensor_list):
        dataset = make_dataset(
            id_dataset=config.DATASET.TYPE,
            config=config.DATASET,
        )
        self.episodes = dataset.episodes
        self.sim_cfg = config.SIMULATOR
        self.sim_cfg.defrost()
        self.sim_cfg.SCENE_DATASET = self.episodes[0].scene_dataset_config
        self.sim_cfg.AGENT_0.SENSORS = sensor_list
        self.sim_cfg.SCENE = self.episodes[0].scene_id
        self.sim_cfg.freeze()
        self.sim = make_sim(
            self.sim_cfg.TYPE, config=self.sim_cfg,
        )
    
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, index):
        self.episode = self.episodes[index]
        self.sim_cfg = merge_sim_episode_config(self.sim_cfg, self.episode)
        self.sim.reconfigure(self.sim_cfg)
        _ = self.sim.reset()

        num_step = np.random.randint(1, self.episode.info["num_action"])
        category = get_category(self.episode)
        location = compute_pointgoal_with_gps_compass(self.sim.get_agent_state(), self.episode)
        image_seq, audio_seq, pose_seq, action_seq = self.get_obs_seq(num_step)
        instruction = np.array(self.episode.instructions)[:, num_step-1]
        x = {
            "image_seq": image_seq,
            "audio_seq": audio_seq,
            "pose_seq": pose_seq,
            "action_seq": action_seq,
            "category": category,
            "location": location,
        }
        y = instruction
        return x, y

    def get_obs_seq(self, num_step):
        image_seq_list = []
        audio_seq_list = []
        pose_seq_list = []
        action_seq_list = []
        oracle_actions = self.sim.get_oracle_actions_from_current_pos()

        for i in range(num_step):
            sim_obs = self.sim._get_sim_observation()
            observations = self.sim._sensor_suite.get_observations(sim_obs)
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
            
            audio = self.sim.get_current_spectrogram_observation(compute_spectrogram)
            pose = compute_pose(self.episode, self.sim.get_agent_state(), i)
            action = oracle_actions[i]

            image_seq_list.append([image])
            audio_seq_list.append([audio])
            pose_seq_list.append([pose])
            action_seq_list.append([action])

            self.sim.step(action)

        image_seq = np.array(image_seq_list)
        audio_seq = np.array(audio_seq_list)
        pose_seq = np.array(pose_seq_list)
        action_seq = np.array(action_seq_list)
    
        return image_seq, audio_seq, pose_seq, action_seq


def compute_spectrogram(audio_data):
    def compute_stft(signal):
        n_fft = 512
        hop_length = 160
        win_length = 400
        stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
        stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
        return stft
    channel1_magnitude = np.log1p(compute_stft(audio_data[0]))
    channel2_magnitude = np.log1p(compute_stft(audio_data[1]))
    spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
    return spectrogram


def quat_to_xy_heading(quat):
    direction_vector = np.array([0, 0, -1])
    heading_vector = quaternion_rotate_vector(quat, direction_vector)
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return np.array([phi], dtype=np.float32)


def compute_pose(episode, agent_state, ep_time):
    origin = np.array(episode.start_position, dtype=np.float32)
    rotation_world_start = quaternion_from_coeff(episode.start_rotation)
    agent_position_xyz = agent_state.position
    rotation_world_agent = agent_state.rotation
    agent_position_xyz = quaternion_rotate_vector(
        rotation_world_start.inverse(), agent_position_xyz - origin
    )
    agent_heading = quat_to_xy_heading(
        rotation_world_agent.inverse() * rotation_world_start
    )
    return np.array(
        [-agent_position_xyz[2], agent_position_xyz[0], agent_heading, ep_time],
        dtype=np.float32
    )


def get_category(episode):
    index = CATEGORY_INDEX_MAPPING[episode.object_category]
    onehot = np.zeros(len(CATEGORY_INDEX_MAPPING.keys()))
    onehot[index] = 1
    return onehot


def compute_pointgoal(source_position, source_rotation, goal_position):
    direction_vector = goal_position - source_position
    direction_vector_agent = quaternion_rotate_vector(
        source_rotation.inverse(), direction_vector
    )
    return np.array(
        [-direction_vector_agent[2], direction_vector_agent[0]],
        dtype=np.float32,
    )


def compute_pointgoal_with_gps_compass(agent_state, episode):
    agent_position = agent_state.position
    rotation_world_agent = agent_state.rotation
    goal_position = np.array(episode.goals[0].position, dtype=np.float32)
    point_goal = compute_pointgoal(
        agent_position, rotation_world_agent, goal_position
    )
    return point_goal


def my_collate_fn(batch):
    batch_size = len(batch)
    max_path_length = 0
    for x, _ in batch:
        if max_path_length < len(x["action_seq"]):
            max_path_length = len(x["action_seq"])
    
    image_shape = np.shape(batch[0][0]["image_seq"][0][0])
    audio_shape = np.shape(batch[0][0]["audio_seq"][0][0])
    
    batched_image_seqs = np.zeros((max_path_length, batch_size) + image_shape, np.float32)
    batched_audio_seqs = np.zeros((max_path_length, batch_size) + audio_shape, np.float32)
    batched_pose_seqs = np.zeros((max_path_length, batch_size, 4), np.float32)
    batched_action_seqs = np.zeros((max_path_length, batch_size, 1), np.float32)
    batched_category = np.zeros((batch_size, 21), np.float32)
    batched_location = np.zeros((batch_size, 2), np.float32)
    path_masks = np.full((batch_size, max_path_length), True)
    seq_lengths = []
    targets = []
    
    for i, (x, y) in enumerate(batch):
        seq_len = len(x["action_seq"])
        path_masks[i, :seq_len] = False
        batched_image_seqs[:seq_len, i] = x["image_seq"][:, 0]
        batched_audio_seqs[:seq_len, i] = x["audio_seq"][:, 0]
        batched_pose_seqs[:seq_len, i] = x["pose_seq"][:, 0]
        batched_action_seqs[:seq_len, i] = x["action_seq"]
        batched_category[i] = x["category"]
        batched_location[i] = x["location"]
        seq_lengths.append(seq_len)
        targets.append(y)
    
    inputs = {
        "rgb": torch.from_numpy(batched_image_seqs[:, :, :, :, :3]).cuda(),
        "depth": torch.from_numpy(batched_image_seqs[:, :, :, :, 3:4]).cuda(),
        "pose": torch.from_numpy(batched_pose_seqs).cuda(),
        "spectrogram": torch.from_numpy(batched_audio_seqs).cuda(),
        "action": torch.from_numpy(batched_action_seqs).cuda(),
        "category": torch.from_numpy(batched_category).cuda(),
        "location": torch.from_numpy(batched_location).cuda(),
        "mask": torch.from_numpy(path_masks).cuda(),
        "seq_lengths": torch.from_numpy(np.array(seq_lengths)).cuda(),
    }
    targets = torch.from_numpy(np.array(targets)).cuda()

    return inputs, targets
