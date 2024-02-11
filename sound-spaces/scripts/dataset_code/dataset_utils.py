import os
import gzip
import json
import numpy as np
import networkx as nx


def save_split_to_disk(dataset_split, split, dataset_dir):
    split_dir = os.path.join(dataset_dir, split)
    content_dir = os.path.join(split_dir, 'content')
    os.makedirs(content_dir, exist_ok=True)

    if split.startswith('val') or split.startswith('test'):
        all_episodes = []
        for dataset in dataset_split['datasets']:
            all_episodes += dataset.episodes
        dataset_to_write = dataset_split['datasets'][0]
        dataset_to_write.episodes = all_episodes
        with gzip.GzipFile(os.path.join(split_dir, '{}.json.gz'.format(split)), 'w') as fo:
            json_str = dataset_to_write.to_json()
            json_bytes = json_str.encode('utf-8')
            fo.write(json_bytes)
    else:
        with gzip.GzipFile(os.path.join(split_dir, '{}.json.gz'.format(split)), 'w') as fo:
            json_str = json.dumps({"episodes": []})
            json_bytes = json_str.encode('utf-8')
            fo.write(json_bytes)

        for i, dataset in enumerate(dataset_split['datasets']):
            json_str = dataset.to_json()
            json_bytes = json_str.encode('utf-8')

            file_path = os.path.join(content_dir, '{}.json.gz'.format(dataset_split['scenes'][i]))
            with gzip.GzipFile(file_path, 'w') as fo:
                fo.write(json_bytes)


def compute_num_action(graph, s, r, rotation):
    orientation = (270 - rotation) % 360
    num_action = 0
    shortest_path = nx.shortest_path(graph, source=r, target=s)

    current_node = r
    for i, next_node in enumerate(shortest_path[1:]):
        current_pos = graph.nodes[current_node]['point']
        next_pos = graph.nodes[next_node]['point']
        direction = np.round(np.rad2deg(np.arctan2(next_pos[2] - current_pos[2],
                                                   next_pos[0] - current_pos[0]))) % 360
        #         print(current_pos, next_pos)
        #         print(orientation, direction)
        angle_distance = abs(direction - orientation) % 180
        angle_distance = angle_distance if angle_distance <= 180 else 360 - angle_distance
        num_action += angle_distance // 90 + 1
        if i != 0:
            assert angle_distance <= 90

        current_node = next_node
        orientation = direction

    num_action += 1
    return int(num_action)


def sample_from_datasets(split, datasets, max_num_episode):
    total_num_episode = sum([len(dataset.episodes) for dataset in datasets])
    print('{} has max number of episodes: {}'.format(split.upper(), max_num_episode))
    remaining_num_episode = max_num_episode
    # uniformly sample max_num_episode episodes from datasets
    for i, dataset in enumerate(datasets):
        if i != len(datasets) - 1:
            num_episode = int(len(dataset.episodes) / total_num_episode * max_num_episode)
            remaining_num_episode = remaining_num_episode - num_episode
        else:
            num_episode = remaining_num_episode

        dataset.episodes = dataset.episodes[:num_episode]
