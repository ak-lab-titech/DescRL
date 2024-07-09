import numpy as np
import lmdb
import torch
from torch.utils.data import Dataset
from tqdm import trange
import msgpack_numpy

from xgenerator.common.lang import tokens2sentences, R2RLang
from xgenerator.common.load_lmdb import EOS_IDX, BOS_IDX, PAD_IDX, d3_40_colors_rgb


class HFR2RDataset(Dataset):
    """
    R2RDataset for HuggingFace.
    """
    def __init__(
        self,
        data_path: str,
        data_num: int,
        max_instruction_length: int,
        prompt: str=None, # TODO
        n_slice: int=None,
        need_action_and_depth_semantic: bool = False,
    ):
        self.data_path = data_path      
        self.data_num = data_num
        self.add_bos = False
        self.max_instruction_length = max_instruction_length

        env = lmdb.open(self.data_path, readonly=True, lock=False)
        r2r_lang = R2RLang("r2r")
        self.image_seqs = []
        self.action_seqs = []
        self.instructions = []

        self.need_action_and_depth_semantic = need_action_and_depth_semantic

        txn = env.begin()
        for index in trange(self.data_num):
            value = txn.get(str(index).encode('latin-1'))
            value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
            observation_seq = dict(value[0])
            
            if self.need_action_and_depth_semantic:
                semantic = np.take(
                    d3_40_colors_rgb,
                    observation_seq["semantic"],
                    axis=0,
                ).astype(np.uint8) / 255.0
                rgb = np.array(observation_seq["rgb"] / 255.0).astype(np.float32)
                depth = np.array(observation_seq["depth"])
                semantic = np.squeeze(semantic, axis=3)
                image_seq = np.concatenate([rgb, depth, semantic], axis=3).astype(np.float32)
                image_seq = torch.from_numpy(image_seq)
            else:
                image_seq = np.array(observation_seq["rgb"]).astype(np.uint8)
                image_seq = torch.from_numpy(image_seq) # (l, h, w, c)
            
            action_seq = torch.from_numpy(np.eye(4)[np.array(value[2])].astype(np.int8))

            if n_slice is not None:
                n_frame = len(image_seq)
                indices = np.arange(0, n_frame, n_frame / n_slice).astype(int)
                image_seq = image_seq[indices]
                action_seq = action_seq[indices]

            self.image_seqs.append(image_seq)
            self.action_seqs.append(action_seq)

            token = np.array(observation_seq["instruction"][0])
            token = token[token != PAD_IDX]
            instruction = tokens2sentences(torch.from_numpy(token).view(-1, 1)[:self.max_instruction_length, :], r2r_lang)
            instruction = "".join(instruction)[:-1] # removing the last space.
            self.instructions.append(instruction)

        txn.commit()
        env.close()
    
    def __len__(self):
        return self.data_num
    
    def __getitem__(self, index):
        if self.need_action_and_depth_semantic:
            return self.image_seqs[index], self.action_seqs[index], self.instructions[index]
        else:
            return self.image_seqs[index], self.instructions[index]


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['multi_modal_projector', 'vision_model']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)