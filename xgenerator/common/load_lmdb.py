import time
import sys
import gzip
import json

import lmdb
import msgpack_numpy
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.utils import try_cuda


def get_lmdb_keys(dir_name: str):
    """
    lmdbデータのキーを全て取得する
    """
    keys = []
    env = lmdb.open(dir_name, readonly=True)
    txn = env.begin()
    cursor = txn.cursor()

    # キーを取得して表示
    for key, _ in cursor:
        # print(key.decode('utf-8'))  # バイト文字列をデコードして表示
        keys.append(key.decode('utf-8'))

    # クリーンアップ
    cursor.close()
    txn.commit()
    env.close()
    return keys

def get_a_lmdb_data(dir_name: str, key: str):
    """
    lmdbのデータを一つ取得する
    """
    env = lmdb.open(dir_name, readonly=True)
    txn = env.begin()

    value = txn.get(key)
    value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
    observation = dict(value[0])
    prev_actions = value[1]
    actions = value[2]
    instr_text = value[3][0]

    txn.commit()
    env.close()
    
    return observation, prev_actions, actions, instr_text

def get_all_lmdb_data(dir_name: str):
    """
    全てのlmdbのデータを取得する
    """
    keys = get_lmdb_keys(dir_name)
    observations = []
    actions = []
    instructions = []
    for key in keys:
        key = key.encode('latin-1')
        obs, _, a, instr = get_a_lmdb_data(dir_name, key)
        observations.append(obs)
        actions.append(a)
        instructions.append(instr)
    return observations, actions, instructions


class R2RDataset(Dataset):
    def __init__(self, data_path: str, use_image_feature: bool):
        self.data_path = data_path
        self.use_image_feature = use_image_feature
        
        self.data_num = None
        self.image_seqs = None
        self.action_seqs = None
        self.instructions = None
        self.load_data()
    
    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        x = {
            "image_seq": self.image_seqs[index],
            "action_seq": self.action_seqs[index],
        }
        y = self.instructions[index]
        return x, y
    
    def load_data(self):
        """
        data_pathのデータを読み込んでinputsとtargetsを作成
        """
        observations, actions, _ = get_all_lmdb_data(self.data_path)
        self.data_num = len(actions)
    
        if not self.use_image_feature:
            self.image_seqs = [
                np.concatenate([obs["rgb"], obs["depth"]], 3) for obs in observations
            ] # (data_num, seq_len, h, w, 4)
        else:
            self.image_seqs = [
                np.concatenate([obs["rgb_features"], obs["depth_features"]], 1) for obs in observations
            ] # (data_num, seq_len, fea_dim, 4, 4)
        self.action_seqs = [np.eye(4)[np.array(action)] for action in actions] # (data_num, seq_len, 4)
        
        self.instructions = [obs["instruction"][0] for obs in observations] # (data_num, instr_len)


def my_collate_fn(batch):
    batch_size = len(batch)
    max_path_length = 0
    for x, _ in batch:
        if max_path_length < len(x["action_seq"]):
            max_path_length = len(x["action_seq"])
    
    batched_image_seqs = [
        np.zeros(
            (batch_size,) + np.shape(batch[0][0]["image_seq"][0]),
            np.float32,
        ) for _ in range(max_path_length)
    ]
    batched_action_seqs = [
        np.zeros(
            (batch_size, 4),
            np.float32,
        ) for _ in range(max_path_length)
    ]
    path_masks = np.full((batch_size, max_path_length), True)
    targets = []
    
    for i, (x, y) in enumerate(batch):
        image_seq = x["image_seq"]
        action_seq = x["action_seq"]
        assert len(image_seq) == len(action_seq)
        path_masks[i, :len(action_seq)] = False
        for t, (image, action) in enumerate(zip(image_seq, action_seq)):
            batched_image_seqs[t][i] = image
            batched_action_seqs[t][i] = action
        
        targets.append(y)
                
    batched_action_seqs = [
        try_cuda(Variable(
            torch.from_numpy(act),
            requires_grad=False,
        )) for act in batched_action_seqs
    ]
    batched_image_seqs = [
        try_cuda(Variable(
            torch.from_numpy(img),
            requires_grad=False,
        )) for img in batched_image_seqs
    ]
    path_masks = try_cuda(torch.from_numpy(path_masks))
    
    inputs = {
        "image_seqs": batched_image_seqs,    # (max_l, (b, 2176, 4, 4))
        "action_seqs": batched_action_seqs,  # (max_l, (b, 4))
        "mask": path_masks,          # (b, max_l)
    }
    targets = try_cuda(torch.from_numpy(np.array(targets))) # (b, 200)
    
    return inputs, targets
    

if __name__=="__main__":
    lmdb_dir = "/home/0/19B30511/av-nav/VLN-CE/data/trajectories_dirs/cma_dagger_gtaction_us100/trajectories.lmdb"
    s = time.time()
    # observations, actions, instructions = get_all_lmdb_data(lmdb_dir)
    print(f"start!")
    dataset = R2RDataset(lmdb_dir, True)
    print(f"define dataset! (time: {time.time() - s} [sec])")
    s = time.time()
    dataloader = DataLoader(
        dataset,
        batch_size=10,
        shuffle=True,
        drop_last=True,
        collate_fn=my_collate_fn,
    )
    print(f"define dataloader! (time: {time.time() - s} [sec])")
    s = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        print(f"--------------------------- i:{i} ----------------------")
        print(f"inputs:")
        print(f"image_seqs: {np.shape(inputs['image_seqs'])}, {np.shape(inputs['image_seqs'][0])}")
        print(f"action_seqs: {np.shape(inputs['action_seqs'])}, {np.shape(inputs['action_seqs'][0])}")
        print(f"mask: {np.shape(inputs['mask'])}")
        print(f"targets: {np.shape(targets)}")
        
    print(f"Time: {time.time() - s} [sec]")
    
    
    
    