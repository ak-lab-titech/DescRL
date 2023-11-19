import time
import sys
import gc

import lmdb
import msgpack_numpy
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

PAD_IDX = 0
BOS_IDX = 2504
EOS_IDX = 2505


def get_lmdb_keys(dir_name: str):
    """
    lmdbデータのキーを全て取得する
    """
    keys = []
    env = lmdb.open(dir_name, readonly=True, lock=False)
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
    env = lmdb.open(dir_name, readonly=True, lock=False)
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
    def __init__(
        self,
        data_path: str,
        use_image_feature: bool,
        data_num: int,
        add_bos: bool,
        skip_frame_per: int,
        max_instruction_length: int,
    ):
        self.data_path = data_path
        self.use_image_feature = use_image_feature        
        self.data_num = data_num
        self.add_bos = add_bos
        self.skip_frame_per = skip_frame_per
        self.max_instruction_length = max_instruction_length
        
        env = lmdb.open(self.data_path, readonly=True, lock=False)
        self.image_seqs = []
        self.action_seqs = []
        self.instructions = []
        txn = env.begin()
        for index in range(self.data_num):
            value = txn.get(str(index).encode('latin-1'))
            value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
            observation_seq = dict(value[0])
            if self.use_image_feature:
                image_seq = np.concatenate(
                    [observation_seq["rgb_features"][::self.skip_frame_per], observation_seq["depth_features"][::self.skip_frame_per]], 1
                ).astype(np.float32)
            else:
                image_seq = np.concatenate(
                    [observation_seq["rgb"][::self.skip_frame_per], observation_seq["depth"][::self.skip_frame_per]], 3
                ).astype(np.float32)
            action_seq = np.eye(4)[np.array(value[2][::self.skip_frame_per])].astype(np.int8)
            instruction = np.array(observation_seq["instruction"][0]).astype(np.uint16)

            del observation_seq
            del value
            gc.collect()

            self.image_seqs.append(image_seq)
            self.action_seqs.append(action_seq)
            self.instructions.append(instruction)
        txn.commit()
        env.close()

    def __len__(self):
        return self.data_num

    def __getitem__(self, index):
        image_seq = self.image_seqs[index]
        action_seq = self.action_seqs[index]
        instruction = self.instructions[index]        
        x = {
            "image_seq": image_seq,
            "action_seq": action_seq,
        }
        y = instruction
        
        if self.add_bos:
            y = np.concatenate([[BOS_IDX], y[:-1]])

        for i in range(len(y)):
            if y[i] == 0:
                y[i] = EOS_IDX
                break
        return x, y[:self.max_instruction_length]


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
    seq_lengths = []
    targets = []
    
    for i, (x, y) in enumerate(batch):
        image_seq = x["image_seq"]
        action_seq = x["action_seq"]
        assert len(image_seq) == len(action_seq)
        path_masks[i, :len(action_seq)] = False
        for t, (image, action) in enumerate(zip(image_seq, action_seq)):
            batched_image_seqs[t][i] = image
            batched_action_seqs[t][i] = action
        seq_lengths.append(len(image_seq))
        targets.append(y)
    
    inputs = {
        "image_seqs": batched_image_seqs,    # (max_l, b, 2176, 4, 4))
        "action_seqs": batched_action_seqs,  # (max_l, b, 4))
        "mask": path_masks,          # (b, max_l)
        "seq_lengths": np.array(seq_lengths),
    }
    # targets: (b, 200)
    return inputs, targets
    

if __name__=="__main__":
    lmdb_dir = "/home/0/19B30511/av-nav/VLN-CE/data/trajectories_dirs/cma_dagger_gtaction_us100/trajectories.lmdb"
    s = time.time()
    # observations, actions, instructions = get_all_lmdb_data(lmdb_dir)
    print(f"start!")
    dataset = R2RDataset(lmdb_dir, True, 100, True, 4, 200)
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
    
    
    
    