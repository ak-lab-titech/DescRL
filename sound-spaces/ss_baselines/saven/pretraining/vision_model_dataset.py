
import os
import pickle

import numpy as np

import torch
# import torch.nn as nn
from torch.utils.data import Dataset


class VisionDataset(Dataset):
    def __init__(self, scenes_nodes, num_objects, num_regions):
        self.scenes_nodes = scenes_nodes
        self.num_objects = num_objects
        self.num_regions = num_regions
        self.files = list()
        self.scene_obs_dir = 'data/scene_observations/mp3d_size128'  # 'data/scene_observations/mp3d', 'data/scene_observations_saven/mp3d'

        for scene in self.scenes_nodes:
            for node_labels in self.scenes_nodes[scene]:
                self.files.append((scene, node_labels))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        scene, node_labels = self.files[item]
        node, objects_id, regions_id = node_labels['node'], node_labels['objects_id'], node_labels['regions_id']

        scene_obs_file = os.path.join(self.scene_obs_dir, '{}.pkl'.format(scene))
        with open(scene_obs_file, 'rb') as fo:
            scene_data = pickle.load(fo)

        rgb_img = scene_data[node]['rgb'][:, :, :3]  # remove alpha channel
        rgb_img = rgb_img / 255.0  # normalize RGB
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        rgb_img = rgb_img.transpose((2, 0, 1))
        rgb_img = torch.from_numpy(rgb_img)

        objects_id = torch.tensor([1 if obj_id in objects_id else 0 for obj_id in range(self.num_objects)])
        regions_id = torch.tensor([1 if reg_id in regions_id else 0 for reg_id in range(self.num_regions)])

        inputs_outputs = (rgb_img, objects_id, regions_id)

        return inputs_outputs
