
import attr
import copy
import pickle
import random
import os
import sys
from collections import defaultdict

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

import gzip
import librosa
from scipy.spatial import cKDTree
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.io import wavfile
from numpy.linalg import norm
import json

from ss_baselines.savi.ppo.policy import Net, AudioNavBaselinePolicy, Policy
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
from habitat.core.dataset import Dataset
from soundspaces.utils import load_metadata
from soundspaces.tasks.semantic_audionav_task import SemanticAudioGoalNavEpisode, SemanticAudioGoal, ObjectViewLocation
from mp3d_utils import HouseReader, SCENE_SPLITS, SOUND_SPLITS
from dataset_utils import compute_num_action


class SemanticDataset(Dataset):
    def __init__(self, scene, sounds_length, shuffle=True):

        self.episodes = []
        self.scene = scene
        scene_meta_dir = os.path.join('../data/metadata/mp3d', scene)
        scene_mesh_dir = os.path.join(scene, '{}.glb'.format(scene))

        # init house reader
        semantic_annotation_file = os.path.join('../data/scene_datasets', 'mp3d', scene, scene + '.house')
        if not os.path.exists(semantic_annotation_file):
            print('Scene {} has no house file'.format(scene))
            return
        house_reader = HouseReader(semantic_annotation_file)

        points, graph = load_metadata(scene_meta_dir)
        nodes = []
        graph_points = []
        for node in graph.nodes():
            nodes.append(node)
            graph_points.append(graph.nodes[node]['point'])

        if len(nodes) == 0:
            print('Scene {} has no graph nodes'.format(scene))
            return
        nodes = np.array(nodes)
        graph_points = np.array(graph_points)
        mesh_points = np.stack([graph_points[:, 0], -graph_points[:, 2], graph_points[:, 1] + 1.5], axis=-1)

        objects = house_reader.find_objects_with_mpcat40_indices()
        if len(objects) == 0:
            print('Scene {} has no object of target categories'.format(scene))
            return

        tol = 1
        for obj in objects:
            sounding_object_name_ = house_reader.category_index2mpcat40_name[obj.category_index]

            # find viewpoints
            object_center_mesh_point = np.array([obj.px, obj.py, obj.pz])
            v = mesh_points - object_center_mesh_point
            a0 = np.array([obj.a0x, obj.a0y, obj.a0z])
            a1 = np.array([obj.a1x, obj.a1y, obj.a1z])
            a2 = np.cross(a0, a1) / np.linalg.norm(np.cross(a0, a1))
            d0 = np.inner(v, a0)
            d1 = np.inner(v, a1)
            d2 = np.inner(v, a2)
            inside_bbx = (abs(d0) < obj.r0 + tol) & (abs(d1) < obj.r1 + tol) & (abs(d2) < obj.r2 + tol)
            inside_bbx_indices = np.nonzero(inside_bbx)[0]
            if len(inside_bbx_indices) == 0:
                continue
            viewpoint_nodes = nodes[inside_bbx_indices]
            viewpoints_graph_points = graph_points[inside_bbx_indices]
            viewpoints_mesh_points = mesh_points[inside_bbx_indices]

            # choose the closest viewpoint to be the sound source
            distance_to_center = np.linalg.norm(viewpoints_mesh_points - object_center_mesh_point)
            min_d_index = np.argmin(distance_to_center)
            s = viewpoint_nodes[min_d_index]

            # remove viewpoints that are not on the same graph as the source node
            same_graph_viewpoint_nodes = []
            same_graph_viewpoints_graph_points = []
            for i, node in enumerate(viewpoint_nodes):
                if node == s or nx.has_path(graph, s, node):
                    same_graph_viewpoint_nodes.append(node)
                    same_graph_viewpoints_graph_points.append(viewpoints_graph_points[i])

            # enumerate all possible receiver positions for this object location and filter no-sound pairs
            for r in graph.nodes():
                if r in same_graph_viewpoint_nodes:
                    continue  # The receiver cannot be at the viewpoint (terminal state)

                try:
                    geodesic_distance_ = nx.shortest_path_length(graph, s, r) * GRID_SIZE
                except nx.exception.NetworkXNoPath:
                    continue

                angle = random.choice([0, 90, 180, 270])
                sr, data = wavfile.read('../data/binaural_rirs/default/{}/{}/{}/{}_{}.wav'.format(scene, angle, r, r, s))
                if data.shape[0] == 0:
                    continue

                # We make this a list of coefficients, because Dataset's episodes cannot handle np.quaternion
                rotation_angle = np.radians(angle)
                agent_rotation = quat_to_coeffs(quat_from_angle_axis(rotation_angle, np.array([0, 1, 0]))).tolist()
                goal_radius = 0.00001
                goal = SemanticAudioGoal(
                    object_id=obj.object_index,
                    object_name=sounding_object_name_,
                    object_category=None,
                    room_id=obj.region_index,
                    room_name=None,
                    view_points=same_graph_viewpoints_graph_points,
                    position=graph.nodes()[s]['point'],
                    radius=goal_radius
                )

                sound_length = sounds_length[sounding_object_name_ + '.wav']
                duration = int(np.random.normal(15, 9, 1)[0])
                duration = np.clip(duration, 5, 500)
                if duration < sound_length:
                    offset = np.random.randint(0, sound_length - duration)
                else:
                    offset = np.random.randint(0, sound_length)

                episode_ = SemanticAudioGoalNavEpisode(
                    goals=[goal],
                    episode_id=str(len(self.episodes)),
                    scene_id=scene_mesh_dir,
                    start_position=graph.nodes()[r]['point'],
                    start_rotation=agent_rotation,
                    info={"geodesic_distance": geodesic_distance_,
                          "num_action": compute_num_action(graph, s, r, int(np.rad2deg(rotation_angle)))},
                    sound_id=os.path.join('sounds_semantic', sounding_object_name_ + '.wav'),
                    object_category=sounding_object_name_,
                    offset=str(offset),
                    duration=str(duration)
                )
                self.episodes.append(episode_)

        if shuffle:
            random.shuffle(self.episodes)
        print('Scene {} has total {} episodes'.format(scene, len(self.episodes)))


def sample_from_datasets(split_, scenes_dataset_split_, max_num_episode):

    total_num_episode_ = sum([len(scenes_dataset_split_[scene].episodes) for scene in scenes_dataset_split_])
    if max_num_episode >= total_num_episode_:
        print("total_num_episode ({}) is less than max_num_episode ({})".format(total_num_episode_, max_num_episode))
        return scenes_dataset_split_

    print('{}: Uniformly sampling {} from {} total episodes'.format(split_.upper(), max_num_episode, total_num_episode_))
    scenes_dataset_split2 = {}
    scene_with_max_episodes = ''
    max_episodes = 0
    # uniformly sample max_num_episode episodes from datasets
    for scene in scenes_dataset_split_:
        dataset_ = copy.deepcopy(scenes_dataset_split_[scene])
        num_episode = int(len(dataset_.episodes) / total_num_episode_ * max_num_episode)
        dataset_.episodes = dataset_.episodes[:num_episode]
        scenes_dataset_split2[scene] = dataset_

        if len(dataset_.episodes) > max_episodes:
            max_episodes = len(dataset_.episodes)
            scene_with_max_episodes = scene

    remaining_num_episode = max_num_episode - total_num_episode_
    if remaining_num_episode:
        scenes_dataset_split2[scene_with_max_episodes].episodes.extend(
            scenes_dataset_split_[scene_with_max_episodes].episodes[-remaining_num_episode:])

    return scenes_dataset_split2


def save_split_to_disk(scenes_dataset_split_, split_, dataset_dir_):

    split_dir = os.path.join(dataset_dir_, split_)
    content_dir = os.path.join(split_dir, 'content')
    os.makedirs(content_dir, exist_ok=True)

    if 'train' not in split_:
        with gzip.GzipFile(os.path.join(split_dir, '{}.json.gz'.format(split_)), 'w') as fo:
            json_str = json.dumps({"episodes": []})
            json_bytes = json_str.encode('utf-8')
            fo.write(json_bytes)

        num_split = 10
        split_count = 1000 // num_split

        all_episodes = []
        for scene in scenes_dataset_split_:
            all_episodes += scenes_dataset_split_[scene].episodes

        dataset_to_write = Dataset()
        for i in range(num_split):
            dataset_to_write.episodes = all_episodes[i * split_count: (i + 1) * split_count]
            with gzip.GzipFile(os.path.join(content_dir, 'split_{}.json.gz'.format(i)), 'w') as fo:
                json_str = dataset_to_write.to_json()
                json_bytes = json_str.encode('utf-8')
                fo.write(json_bytes)
    else:
        with gzip.GzipFile(os.path.join(split_dir, '{}.json.gz'.format(split_)), 'w') as fo:
            json_str = json.dumps({"episodes": []})
            json_bytes = json_str.encode('utf-8')
            fo.write(json_bytes)

        for scene in scenes_dataset_split_:
            json_str = scenes_dataset_split_[scene].to_json()
            json_bytes = json_str.encode('utf-8')

            file_path = os.path.join(content_dir, '{}.json.gz'.format(scene))
            with gzip.GzipFile(file_path, 'w') as fo:
                fo.write(json_bytes)


if __name__ == '__main__':

    print("Current working directory: {0}".format(os.getcwd()))
    os.chdir('../../../sound-spaces/scripts')
    print("Current working directory: {0}".format(os.getcwd()))

    dataset_dir = '../data/datasets/semantic_aven/mp3d/v1'
    mp3d_scenes_dataset_filepath = dataset_dir + os.sep + "mp3d_scenes_dataset.bin"
    source_sound_dir = '../data/sounds_semantic'

    GRID_SIZE = 1.0
    MIN_DISTANCE = 4
    RATIO_THRESHOLD = 1.1
    MAX_TEST_EPISODE = 1000

    SCENES = set([scene for split_scenes in SCENE_SPLITS.values() for scene in split_scenes])
    print("scenes: ", len(SCENES), SCENES)

    # rir_sampling_rate = 16000
    # SOUNDS_LENGTH = {}
    # for sound_file in os.listdir(source_sound_dir):
    #     sound_data, sr = librosa.load(os.path.join(source_sound_dir, sound_file), sr=rir_sampling_rate)
    #     SOUNDS_LENGTH[sound_file] = sound_data.shape[0] // rir_sampling_rate
    SOUNDS_LENGTH = {'bed.wav': 112, 'stool.wav': 90, 'clothes.wav': 37, 'fireplace.wav': 94, 'table.wav': 48,
                     'tv_monitor.wav': 60, 'plant.wav': 74, 'chair.wav': 48, 'bathtub.wav': 139,
                     'gym_equipment.wav': 180, 'chest_of_drawers.wav': 38, 'shower.wav': 64, 'towel.wav': 46,
                     'toilet.wav': 154, 'picture.wav': 44, 'cabinet.wav': 53, 'cushion.wav': 45, 'seating.wav': 78,
                     'sink.wav': 60, 'sofa.wav': 42, 'counter.wav': 69}
    print("SOUNDS_LENGTH: ", len(SOUNDS_LENGTH), SOUNDS_LENGTH)

    SPLITS = {}
    for scene_split_type in ['seen-scenes', 'unseen-scenes']:
        for sound_split_type in ['heard-sounds', 'unheard-sounds']:

            scene_split = SCENE_SPLITS['test']
            if scene_split_type == 'seen-scenes':
                scene_split = SCENE_SPLITS['train']

            sound_split = SOUND_SPLITS['test']
            if sound_split_type == 'heard-sounds':
                sound_split = SOUND_SPLITS['train']

            train_or_test = 'train' if scene_split_type == 'seen-scenes' and sound_split_type == 'heard-sounds' else 'test'
            SPLITS['_'.join([train_or_test, scene_split_type, sound_split_type])] = {'scenes': scene_split, 'sounds': sound_split}

    if not os.path.exists(mp3d_scenes_dataset_filepath):
        SCENES_DATASET = {}
        for scene in SCENES:
            print("scene: ", scene)
            SCENES_DATASET[scene] = SemanticDataset(scene, SOUNDS_LENGTH)

        os.makedirs(dataset_dir, exist_ok=True)
        output_file = open(mp3d_scenes_dataset_filepath, "wb")
        pickle.dump(SCENES_DATASET, output_file)
        output_file.close()
    else:
        print("Loading mp3d_scenes_dataset")
        bin_file = open(mp3d_scenes_dataset_filepath, "rb")
        SCENES_DATASET = pickle.load(bin_file)
        bin_file.close()

    for split in SPLITS:
        print("split: ", split)
        scenes_dataset_split = {}
        scenes = SPLITS[split]['scenes']
        sounds = SPLITS[split]['sounds']

        for scene in scenes:
            # if scene not in SCENES_DATASET:
            #     continue
            dataset = copy.deepcopy(SCENES_DATASET[scene])
            filtered_episodes = []
            for episode in dataset.episodes:
                sounding_object_name = episode.object_category

                if sounding_object_name + '.wav' not in sounds:
                    continue

                geodesic_distance = episode.info['geodesic_distance']
                euclidean_distance = norm(np.array(episode.start_position) - np.array(episode.goals[0].position))
                ratio = geodesic_distance / euclidean_distance

                # if geodesic_distance < 6 or geodesic_distance > 16 or ratio < RATIO_THRESHOLD:
                if geodesic_distance < MIN_DISTANCE or ratio < RATIO_THRESHOLD:
                    continue
                else:
                    # episode.duration = int(episode.info['num_action'] * 0.7)
                    filtered_episodes.append(episode)

            dataset.episodes = filtered_episodes
            scenes_dataset_split[scene] = dataset

            print('After filtering {}, number of episodes reduce from {} to {}'.format(scene,
                                                                                       len(SCENES_DATASET[scene].episodes),
                                                                                       len(filtered_episodes)))

        if 'train' not in split:
            scenes_dataset_split = sample_from_datasets(split, scenes_dataset_split, MAX_TEST_EPISODE)
        else:
            total_num_episode = sum([len(scenes_dataset_split[scene].episodes) for scene in scenes_dataset_split])
            print('{}: has {} total episodes'.format(split.upper(), total_num_episode))

        save_split_to_disk(scenes_dataset_split, split, dataset_dir)
