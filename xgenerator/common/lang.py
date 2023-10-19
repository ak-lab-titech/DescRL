import sys
import gzip
import json

import numpy as np

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.load_lmdb import get_all_lmdb_data


class R2RLang:
    def __init__(self, name: str):
        self.name = name
        # self.word2index = {}
        # self.word2count = {}

        with gzip.open(
            "/home/0/19B30511/av-nav/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
            self.index2word = data["instruction_vocab"]["word2idx_dict"]
            self.vocab_size = data["instruction_vocab"]["num_vocab"]

        with gzip.open(
            "/home/0/19B30511/av-nav/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings.json.gz",
            'rt',
            encoding='utf-8',
        ) as f:
            self.glove_vec = np.array(json.load(f))
