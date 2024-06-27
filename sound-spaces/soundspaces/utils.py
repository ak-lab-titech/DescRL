# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import pickle

import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from habitat_sim.utils.common import d3_40_colors_rgb


def load_metadata(parent_folder):
    points_file = os.path.join(parent_folder, 'points.txt')
    if "replica" in parent_folder:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5528907,
            -points_data[:, 2])
        )
    else:
        graph_file = os.path.join(parent_folder, 'graph.pkl')
        points_data = np.loadtxt(points_file, delimiter="\t")
        points = list(zip(
            points_data[:, 1],
            points_data[:, 3] - 1.5,
            -points_data[:, 2])
        )
    if not os.path.exists(graph_file):
        raise FileExistsError(graph_file + ' does not exist!')
    else:
        with open(graph_file, 'rb') as fo:
            graph = pickle.load(fo)

    return points, graph


def _to_tensor(v):
    if torch.is_tensor(v):
        return v
    elif isinstance(v, np.ndarray):
        return torch.from_numpy(v)
    else:
        return torch.tensor(v, dtype=torch.float)


def convert_semantic_object_to_rgb(x):
    semantic_img = Image.new("P", (x.shape[1], x.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((x.flatten() % 40).astype(np.uint8))
    semantic_img = np.array(semantic_img.convert("RGB"))
    return semantic_img


def generate_video(image_seq: torch.Tensor, output_file: str, frame_rate: int=5):
    """
    image_seq: (seq_len, 1, H, W, C)
    """
    if image_seq.shape[4] == 7:
        frame_size = (image_seq.shape[2]*3, image_seq.shape[3])
    elif image_seq.shape[4] == 4:
        frame_size = (image_seq.shape[2]*2, image_seq.shape[3])
    else:
        frame_size = (image_seq.shape[2], image_seq.shape[3])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    for i in range(len(image_seq)):
        rgb = image_seq[i, 0, :, :, :3].to('cpu').detach().numpy().copy() * 255
        if image_seq.shape[4] > 3:
            depth = image_seq[i, 0, :, :, 3].to('cpu').detach().numpy().copy()
            depth = np.uint8(cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX))
            depth = np.array(cv2.applyColorMap(depth, cv2.COLORMAP_JET))
            if image_seq.shape[4] == 7:
                semantic = image_seq[i, 0, :, :, 4:].to('cpu').detach().numpy().copy() * 255
                frame = np.concatenate([rgb, depth, semantic], axis=1)
            else:
                frame = np.concatenate([rgb, depth], axis=1)
        else:
            frame = rgb
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()


def visualize_spectrogram(spectrogram, output_file):
    """
    spectrogram: (H, W, 2)
    """
    left_channel = spectrogram[:, :, 0]
    right_channel = spectrogram[:, :, 1]

    plt.figure(figsize=(20, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(left_channel, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Left Channel Spectrogram')

    plt.subplot(1, 2, 2)
    plt.imshow(right_channel, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(label='Intensity')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title('Right Channel Spectrogram')

    plt.tight_layout()
    plt.savefig(output_file)