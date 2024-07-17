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
from videollama2.mm_utils import get_model_name_from_path
from videollama2.model.builder import load_pretrained_model
from videollama2.train import process_video
from videollama2.constants import MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token


class IPRLPretrainingLMDBDataset(Dataset):
    def __init__(self, config, split, lmdb_dataset_path, data_num, foundation_model_type=None, foundation_model_path=None, num_frames=16):
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

        self.foundation_model_type = foundation_model_type
        if foundation_model_type is None:
            self.foundation_model = None
        elif foundation_model_type == "video_llama2":
            tokenizer, self.foundation_model, self.processor, _ = load_pretrained_model(
                foundation_model_path, None, get_model_name_from_path(foundation_model_path),
            )
            self.visual_encoder = self.foundation_model.get_model().get_vision_tower().vision_tower
            prompt = "[INST] <<SYS>>\n" \
                "A chat between a curious user and an artificial intelligence assistant." \
                "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                "\n<</SYS>>\n\n <video>\nWhat is the camera wearer doing? [/INST]"
            self.input_ids = tokenizer_MMODAL_token(
                prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors='pt',
            ).unsqueeze(0).to(device="cuda")
            self.num_frames = num_frames
        else:
            raise Exception(f"foundation_model_type: {foundation_model_type}")
    
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
        if self.foundation_model_type is None:
            instruction = np.array(self.episodes[index].instructions)[:, int(step)] # (40,)
        elif self.foundation_model_type == "video_llama2":
            tmp_image_seq = (image_seq.copy().squeeze(1)[:, :, :, :3] * 256).astype(np.uint8)
            visual_tensor = process_video(
                tmp_image_seq.copy(),
                self.processor,
                "pad",
                len(tmp_image_seq),
            ).to(
                dtype=torch.float16,
                device='cuda',
                non_blocking=True,
            ) # (l, c, h, w)
            with torch.no_grad():
                visual_features = self.visual_encoder(
                    visual_tensor,
                    output_hidden_states=True,
                ).pooler_output # (l, 1024)

            image_seq_len = len(tmp_image_seq)
            indices = np.arange(0, image_seq_len,  image_seq_len / self.num_frames).astype(int)
            visual_tensor = process_video(
                tmp_image_seq[indices] if self.num_frames < image_seq_len else tmp_image_seq,
                self.processor,
                "pad",
                self.num_frames if self.num_frames < image_seq_len else image_seq_len,
            ).to(
                dtype=torch.float16,
                device='cuda',
                non_blocking=True,
            ) # (l, c, h, w)
            with torch.inference_mode():
                outputs = self.foundation_model.generate(
                    self.input_ids,
                    images_or_videos=[visual_tensor],
                    modal_list=['video'],
                    do_sample=True,
                    temperature=0.2,
                    # max_new_tokens=1024,
                    max_new_tokens=40,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            logits = torch.stack(outputs.scores).squeeze(1) # (instr_len, vocab_size)
        else:
            raise Exception(f"foundation_model_type: {self.foundation_model_type}")
        
        x = {
            "image_seq": image_seq,
            "audio_seq": audio_seq,
            "pose_seq": pose_seq,
            "action_seq": action_seq,
            "category": category,
            "location": location,
        }
        if self.foundation_model is None:
            return x, instruction
        else:
            return x, visual_features, logits

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
        else:
            raise Exception(f"self.foundation_model_type: {self.foundation_model_type}")
        
        image_seq = x["image_seq"]
        audio_seq = x["audio_seq"]
        pose_seq = x["pose_seq"]
        action_seq = x["action_seq"]
        category = x["category"]
        location = x["location"]        
        
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

    dataset_name = "val_w_past_instruction"
    lmdb_dataset_path = "./data/lmdb_dataset/iprl_pretrain/val"
    save_video_path = "./data/videos/offpolicy_lmdb_dataset/hoge"
    foundation_model_type = "video_llama2"
    foundation_model_path = "DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base"
    num_frames = 16
    indices = [0, 1, 2, 3]
    config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml")
    dataset = IPRLPretrainingLMDBDataset(
        config=config.TASK_CONFIG,
        split=dataset_name,
        lmdb_dataset_path=lmdb_dataset_path,
        data_num=100,
        foundation_model_type=foundation_model_type,
        foundation_model_path=foundation_model_path,
        num_frames=num_frames,
    )

    for idx in indices:
        dataset.visualize_data(idx, f"{save_video_path}/index_{idx}")

