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
from xgenerator.common.lang import R2RLang, tokens2sentences, VideoLLaMA2Lang


class IPRLPretrainingLMDBDataset(Dataset):
    def __init__(
        self,
        config,
        split,
        lmdb_dataset_path,
        data_num,
        foundation_model_type=None,
        fm_lmdb_dataset_path=None,
        environment_type=None,
        device="cuda",
    ):
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
        self.env = lmdb.open(lmdb_dataset_path, readonly=True, lock=False, map_size=5 * 1.1e12)

        config.defrost()
        config.DATASET.SPLIT = tmp_split
        config.freeze()

        self.environment_type = environment_type

        self.foundation_model_type = foundation_model_type
        if foundation_model_type is None:
            self.fm_env = None
        elif foundation_model_type == "video_llama2" or foundation_model_type == "qwen25vl":
            self.fm_env = lmdb.open(fm_lmdb_dataset_path, readonly=True, lock=False, map_size=5 * 1.1e12)
        else:
            raise Exception(f"foundation_model_type: {foundation_model_type}")
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):
        value = self.get_value(index)
        
        if self.environment_type == "ss1-savi":
            image_seq = value[0]   # (seq_len, 1, image_shape)
            audio_seq = value[1]   # (seq_len, 1, spectrogram_shape)
            pose_seq = value[2]    # (seq_len, 1, 4)
            action_seq = value[3]  # (seq_len, 1)
            category = value[4]    # (21,)
            location = value[5]    # (2,)
            objectgoal = None
        elif self.environment_type == "habitat-objnav":
            image_seq = value[0]
            pose_seq = value[1]
            action_seq = value[2]
            objectgoal = value[3]
            audio_seq = None
            location = None
            category = None

        step = pose_seq[-1, 0, 3]
        if self.foundation_model_type is None:
            instruction = np.array(self.episodes[index].instructions)[:, int(step)] # (40,)
        elif self.foundation_model_type == "video_llama2":
            with self.fm_env.begin() as txn:
                fm_value = txn.get(str(index).encode('latin-1'))
            fm_value = msgpack_numpy.unpackb(fm_value, object_hook=msgpack_numpy.decode)
            visual_features, logits = torch.from_numpy(fm_value[0]), torch.from_numpy(fm_value[1])
        elif self.foundation_model_type == "qwen25vl":
            with self.fm_env.begin() as txn:
                fm_value = txn.get(str(index).encode('latin-1'))
            fm_value = msgpack_numpy.unpackb(fm_value, object_hook=msgpack_numpy.decode)
            logits = torch.from_numpy(fm_value[0])
        else:
            raise Exception(f"foundation_model_type: {self.foundation_model_type}")
        
        x = {
            "image_seq": image_seq,
            "audio_seq": audio_seq,
            "pose_seq": pose_seq,
            "action_seq": action_seq,
            "category": category,
            "location": location,
            "objectgoal": objectgoal,
        }
        if self.foundation_model_type is None:
            return x, instruction
        elif self.foundation_model_type == "qwen25vl":
            return x, logits
        elif self.foundation_model_type == "video_llama2":
            return x, visual_features, logits
        else:
            raise Exception()

    def get_value(self, index):
        with self.env.begin() as txn:
            value = txn.get(str(index).encode('latin-1'))
            value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
        return value

    def visualize_data(self, index, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/spectrograms", exist_ok=True)

        if self.foundation_model_type is None:
            x, instruction = self[index]
            words = tokens2sentences(torch.from_numpy(instruction.reshape(-1, 1)), R2RLang("r2r_lang"))
        elif self.foundation_model_type == "video_llama2":
            x, _, logits = self[index]
            instruction = torch.argmax(logits, dim=1) # (instr_len,)
            words = tokens2sentences(instruction.view(-1, 1), VideoLLaMA2Lang("videollama2"))
        elif self.foundation_model_type == "qwen25vl":
            raise NotImplementedError()
        else:
            raise Exception(f"self.foundation_model_type: {self.foundation_model_type}")
        
        image_seq = x["image_seq"]
        audio_seq = x["audio_seq"]
        pose_seq = x["pose_seq"]
        action_seq = x["action_seq"]
        category = x["category"]
        location = x["location"]
        objectgoal = x["objectgoal"]    
        
        f = open(f"{output_dir}/output.txt", "w")
        f.write(f"0: found, 1: forward, 2: left, 3: right\n")
        f.write(f"action_seq: {action_seq}\n")
        f.write(f"pose_seq: {pose_seq}\n")
        f.write(f"instructions: {words}\n")
        f.write(f"category: {category}\n")
        f.write(f"location: {location}\n")
        f.write(f"objectgoal: {objectgoal}\n")
        f.close()

        generate_video(torch.from_numpy(image_seq.copy()), f"{output_dir}/image_seq.mp4")

        for i in range(len(audio_seq)):
            visualize_spectrogram(audio_seq[i][0], f"{output_dir}/spectrograms/spectrogram_{i}.png")


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"start iprl_pretraining_lmdb_dataset!\n")
    f.close()

    dataset_name = "val_w_past_instruction"
    lmdb_dataset_path = "./data/lmdb_dataset/iprl_pretrain/val"
    save_video_path = "./data/videos/offpolicy_lmdb_dataset/val_CNNTF_xgen"
    foundation_model_type = None
    fm_lmdb_dataset_path = None
    indices = [0, 100, 200, 300, 400]
    environment_type = "ss1-savi"
    
    config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml")
    dataset = IPRLPretrainingLMDBDataset(
        config=config.TASK_CONFIG,
        split=dataset_name,
        lmdb_dataset_path=lmdb_dataset_path,
        data_num=100,
        foundation_model_type=foundation_model_type,
        fm_lmdb_dataset_path=fm_lmdb_dataset_path,
        environment_type=environment_type,
    )

    for idx in indices:
        dataset.visualize_data(idx, f"{save_video_path}/index_{idx}")

