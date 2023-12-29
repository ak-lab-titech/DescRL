import sys
import os
import time

import lmdb
import msgpack_numpy
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("/home/0/19B30511/av-nav/myss")
sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")

from soundspaces.utils import generate_video, visualize_spectrogram
from xgenerator.common.lang import R2RLang, tokens2sentences


class IPRLPretrainingLMDBDataset(Dataset):
    def __init__(self, lmdb_dataset_path, data_num):
        self.data_num = data_num
        self.env = lmdb.open(lmdb_dataset_path, readonly=True, lock=False)
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):
        value = self.get_value(index)
        
        image_seq = value[0] # (seq_len, 1, image_shape)
        audio_seq = value[1] # (seq_len, 1, image_sape)
        pose_seq = value[2]  # (seq_len, 1, image_shape)
        action_seq = value[3]
        category = value[4]
        location = value[5]
        instruction = value[6]

        for i in range(len(value)):
            f = open("debug.txt", "a")
            f.write(f"{i}, {np.shape(value[i])}\n")
            f.close()
        
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

    def get_value(self, index):
        with self.env.begin() as txn:
            value = txn.get(str(index).encode('latin-1'))
            value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
        return value

    def visualize_data(self, index, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/spectrograms", exist_ok=True)
        
        value = self.get_value(index)

        image_seq = value[0]
        audio_seq = value[1]
        pose_seq = value[2]
        action_seq = value[3]
        category = value[4]
        location = value[5]
        instruction = value[6]

        words = tokens2sentences(instruction.reshape(-1, 1), R2RLang("r2r_lang"))
        
        f = open(f"{output_dir}/output.txt", "w")
        f.write(f"0: found, 1: forward, 2: left, 3: right\n")
        f.write(f"action_seq: {action_seq}\n")
        f.write(f"pose_seq: {pose_seq}\n")
        f.write(f"instructions: {words}\n")
        f.write(f"category: {category}\n")
        f.write(f"location: {location}\n")
        f.close()

        generate_video(torch.from_numpy(image_seq.copy()), f"{output_dir}/image_seq.mp4")

        for i in range(len(audio_seq)):
            visualize_spectrogram(audio_seq[i][0], f"{output_dir}/spectrograms/spectrogram_{i}.png")


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"start iprl_pretraining_lmdb_dataset!\n")
    f.close()
    dataset = IPRLPretrainingLMDBDataset(
        "./data/lmdb_dataset/lmdb_test", 100
    )
    for i in range(len(dataset)):
        f = open("debug.txt", "a")
        f.write(f"-------- i: {i} ---------\n")
        f.close()
        x, y = dataset[i]
    
    dataset.visualize_data(0, "./data/videos/offpolicy_lmdb_dataset_0")

