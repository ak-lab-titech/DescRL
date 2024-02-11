# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List, Optional
from abc import ABC
import os
import argparse
import logging
import pickle
from collections import defaultdict
import sys

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import PIL.Image
from matplotlib.cm import get_cmap

from habitat_sim.utils.common import quat_to_angle_axis, quat_to_coeffs, quat_from_angle_axis, quat_from_coeffs
from habitat.tasks.nav.nav import NavigationEpisode, NavigationGoal, ShortestPathPoint

from ss_baselines.savi.ppo.policy import Net, AudioNavBaselinePolicy, Policy
from soundspaces.tasks.audionav_task import merge_sim_episode_config
from soundspaces.utils import load_metadata
from soundspaces.simulator import SoundSpacesSim
from ss_baselines.av_nav.config import get_config


def plot_semantic_image_helper(semantic_ids, id_name, semantic_type, output_path, filename, objects_of_interest=None):

    color_map = "gist_ncar"
    cmap = get_cmap(color_map)

    region = False
    if semantic_type == 'region' and objects_of_interest:
        semantic_type = 'object'
        region = True

    plt_handlers = []
    plt_titles = []
    for label_id in sorted(id_name[semantic_type]):
        label_name = id_name[semantic_type][label_id]

        if label_id not in semantic_ids[semantic_type]:
            continue

        if objects_of_interest:
            if semantic_type == 'object' and (label_name not in objects_of_interest):
                semantic_ids['object'] = np.where(semantic_ids['object'] == label_id, 0, semantic_ids['object'])
                continue

        rgba = cmap(label_id / len(id_name[semantic_type]))
        p = plt.Rectangle((0, 0), 1, 1, fc=rgba)
        plt_handlers.append(p)
        plt_titles.append('{value}: {name}'.format(value=label_id, name=label_name))

    if region:
        semantic_ids['region'] = np.where(semantic_ids['object'] != 0, semantic_ids['region'], 0)
        semantic_type = 'region'

        plt_handlers = []
        plt_titles = []
        for label_id in sorted(id_name[semantic_type]):
            if label_id != 0 and label_id in semantic_ids[semantic_type]:
                label_name = id_name[semantic_type][label_id]

                rgba = cmap(label_id / len(id_name[semantic_type]))
                p = plt.Rectangle((0, 0), 1, 1, fc=rgba)
                plt_handlers.append(p)
                count = (semantic_ids[semantic_type] == label_id).sum()  # Count occurrence of label_id
                proportion = round(count / semantic_ids[semantic_type].size, 2)
                plt_titles.append('{value}: {name}, {proportion}'.format(value=label_id, name=label_name,
                                                                         proportion=proportion))

    plt.imshow(semantic_ids[semantic_type], cmap=color_map, vmin=0, vmax=len(id_name[semantic_type]))
    plt.legend(plt_handlers, plt_titles, loc='lower right', bbox_to_anchor=(2, 0.0))  # x, y
    plt.colorbar()
    ooi = "_OOI" if objects_of_interest else ""
    plt.savefig(output_path + os.sep + filename + "_" + semantic_type + "_semantic" + ooi + ".png", bbox_inches='tight',
                dpi=100)
    plt.close()


def plot_semantic_image(semantic_ids, id_name, output_path, filename, all_objects=False, all_regions=False,
                        objects_of_interest=None, ooi_objects=False, ooi_regions=False):

    if ooi_objects or ooi_regions:
        assert objects_of_interest, "objects_of_interest is needed to plot ooi_objects and ooi_regions"

    if all_objects:
        plot_semantic_image_helper(semantic_ids, id_name, 'object', output_path, filename)
    if all_regions:
        plot_semantic_image_helper(semantic_ids, id_name, 'region', output_path, filename)
    if ooi_objects:
        plot_semantic_image_helper(semantic_ids, id_name, 'object', output_path, filename, objects_of_interest)
    if ooi_regions:
        plot_semantic_image_helper(semantic_ids, id_name, 'region', output_path, filename, objects_of_interest)


class Sim(SoundSpacesSim):

    def step(self, action):
        sim_obs = self._sim.get_sensor_observations()

        semantic = sim_obs['semantic']
        scene = self._sim.semantic_scene

        object_instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        object_mapping = np.array(
            [object_instance_id_to_label_id[i] for i in range(len(object_instance_id_to_label_id))])
        object_semantic_ids = np.take(object_mapping, semantic)

        region_instance_id_to_label_id = {}
        for obj in scene.objects:
            if obj.region is None:
                region_instance_id_to_label_id[int(obj.id.split("_")[-1])] = 0
            else:
                region_instance_id_to_label_id[int(obj.id.split("_")[-1])] = obj.region.category.index()

        region_mapping = np.array(
            [region_instance_id_to_label_id[i] for i in range(len(region_instance_id_to_label_id))])
        region_semantic_ids = np.take(region_mapping, semantic)

        del sim_obs['semantic']
        sim_obs['object_semantic'] = object_semantic_ids
        sim_obs['region_semantic'] = region_semantic_ids

        return sim_obs, self._rotation_angle


def main(dataset):

    print("Current working directory: {0}".format(os.getcwd()))
    os.chdir('../sound-spaces')
    print("Current working directory: {0}".format(os.getcwd()))

    mp3d_metadata_filepath = r"data/metadata/mp3d_scenes_semantic_data.bin"
    bin_file = open(mp3d_metadata_filepath, "rb")
    scenes = pickle.load(bin_file)
    objects_id_name = pickle.load(bin_file)
    regions_id_name = pickle.load(bin_file)
    bin_file.close()
    id_name = {'object': objects_id_name, 'region': regions_id_name}

    mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
    bin_file = open(mp3d_objects_of_interest_filepath, "rb")
    ooi_objects_id_name = pickle.load(bin_file)
    ooi_regions_id_name = pickle.load(bin_file)
    bin_file.close()
    objects_of_interest = list(ooi_objects_id_name.values())  # Semantic AVN's 21 objects

    save_obs_images = False
    image_size = 128

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-path",
        type=str,
        default='ss_baselines/av_nav/config/audionav/{}/train_telephone/pointgoal_rgb.yaml'.format(dataset)
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    config = get_config(args.config_path, opts=args.opts)
    config.defrost()
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "SEMANTIC_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT = image_size
    config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = image_size
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT = image_size
    config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH = image_size
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.WIDTH = image_size
    config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = image_size
    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
    config.freeze()
    simulator = None
    scene_obs = defaultdict(dict)
    num_obs = 0
    scene_obs_dir = 'data/scene_observations_saven/' + dataset
    os.makedirs(scene_obs_dir, exist_ok=True)
    metadata_dir = 'data/metadata/' + dataset
    for scene in os.listdir(metadata_dir):
        scene_obs = dict()
        scene_metadata_dir = os.path.join(metadata_dir, scene)
        points, graph = load_metadata(scene_metadata_dir)
        if dataset == 'replica':
            scene_mesh_dir = os.path.join('data/scene_datasets', dataset, scene, 'habitat/mesh_semantic.ply')
        else:
            scene_mesh_dir = os.path.join('data/scene_datasets', dataset, scene, scene + '.glb')

        if save_obs_images:
            scene_obs_img_scene_dir = scene_obs_dir + '_images' + os.sep + scene
            os.makedirs(scene_obs_img_scene_dir, exist_ok=True)

        for node in graph.nodes():
            agent_position = graph.nodes()[node]['point']
            for angle in [0, 90, 180, 270]:
                agent_rotation = quat_to_coeffs(quat_from_angle_axis(np.deg2rad(angle), np.array([0, 1, 0]))).tolist()
                goal_radius = 0.00001
                goal = NavigationGoal(
                    position=agent_position,
                    radius=goal_radius
                )
                episode = NavigationEpisode(
                    goals=[goal],
                    episode_id=str(0),
                    scene_id=scene_mesh_dir,
                    start_position=agent_position,
                    start_rotation=agent_rotation,
                    info={'sound': 'telephone'}
                )

                episode_sim_config = merge_sim_episode_config(config.TASK_CONFIG.SIMULATOR, episode)
                if simulator is None:
                    simulator = Sim(episode_sim_config)
                simulator.reconfigure(episode_sim_config)

                obs, rotation_index = simulator.step(None)

                if save_obs_images:
                    image_filename = str(node) + '_' + str(angle)
                    semantic_ids = {'object': obs['object_semantic'], 'region': obs['region_semantic']}

                    matplotlib.image.imsave(scene_obs_img_scene_dir + os.sep + image_filename + '_rgb.png', obs['rgb'])
                    matplotlib.image.imsave(scene_obs_img_scene_dir + os.sep + image_filename + '_depth.png', obs['depth'])

                    plot_semantic_image(semantic_ids, id_name, scene_obs_img_scene_dir, image_filename, all_objects=True,
                                        all_regions=True, objects_of_interest=objects_of_interest, ooi_objects=True,
                                        ooi_regions=True)

                scene_obs[(node, rotation_index)] = obs
                num_obs += 1

        print('Total number of observations: {}'.format(num_obs))
        with open(os.path.join(scene_obs_dir, '{}.pkl'.format(scene)), 'wb') as fo:
            pickle.dump(scene_obs, fo)
    simulator.close()
    del simulator


if __name__ == '__main__':
    # print('Caching Replica observations ...')
    # main('replica')
    print('Caching Matterport3D observations ...')
    main('mp3d')
