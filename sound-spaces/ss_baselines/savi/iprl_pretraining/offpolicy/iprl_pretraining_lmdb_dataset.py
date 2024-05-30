import sys
import os
import time

import lmdb
import msgpack_numpy
import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.append("/home/4/ud02274/navigation/myss")
sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

from habitat.datasets import make_dataset
from ss_baselines.savi.config.default import get_config
from soundspaces.utils import generate_video, visualize_spectrogram
from xgenerator.common.lang import R2RLang, tokens2sentences


class IPRLPretrainingLMDBDataset(Dataset):
    def __init__(self, config, split, lmdb_dataset_path, data_num):
        tmp_split = config.DATASET.SPLIT
        config.defrost()
        config.DATASET.SPLIT = split
        config.freeze()
        dataset = make_dataset(
            id_dataset=config.DATASET.TYPE,
            config=config.DATASET,
        )
        self.episodes = dataset.episodes
        self.data_num = data_num
        # if "iprl_pretrain_train" in lmdb_dataset_path:
        #     if int(os.environ["RANK"]) % 2 == 0:
        #         lmdb_dataset_path = f"{lmdb_dataset_path}_2"
        #     else:
        #         lmdb_dataset_path = f"{lmdb_dataset_path}"
        self.env = lmdb.open(lmdb_dataset_path, readonly=True, lock=False)

        config.defrost()
        config.DATASET.SPLIT = tmp_split
        config.freeze()
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):
        value = self.get_value(index)
        
        image_seq = value[0]   # (seq_len, 1, image_shape)
        audio_seq = value[1]   # (seq_len, 1, spectrogram_shape)
        pose_seq = value[2]    # (seq_len, 1, 4)
        action_seq = value[3]  # (seq_len, 1)
        category = value[4]    # (21,)
        location = value[5]    # (2,)

        step = pose_seq[-1, 0, 3]
        instruction = np.array(self.episodes[index].instructions)[:, int(step)] # (40,)
        
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
        
        step = pose_seq[-1, 0, 3]
        instruction = np.array(self.episodes[index].instructions)[:, int(step)]

        words = tokens2sentences(torch.from_numpy(instruction.reshape(-1, 1)), R2RLang("r2r_lang"))
        
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
    type_of_dataset = "test" # "train", "val", "test"
    future_or_past = "past"
    indices = [0, 1, 2, 3]


    config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml")

    dataset = IPRLPretrainingLMDBDataset(
        config.TASK_CONFIG,
        f"{type_of_dataset}_w_instruction" if future_or_past == "future" else f"{type_of_dataset}_w_{future_or_past}_instruction",
        f"./data/lmdb_dataset/iprl_pretrain/{type_of_dataset}",
        100,
    )
    for i in range(len(dataset)):
        x, y = dataset[i]
    
    save_path = f"./data/videos/offpolicy_lmdb_dataset/{future_or_past}_{type_of_dataset}_dataset"
    for idx in indices:
        dataset.visualize_data(idx, f"{save_path}/index_{idx}")
