import sys
import gzip
import json

import numpy as np

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.load_lmdb import get_all_lmdb_data


def tokens2sentences(tokens, lang):
    """
    tokens' shape must be (sentence_len, batch).
    """
    sentences = ["" for _ in range(len(tokens[0]))]
    for token in tokens:
        token = token.to('cpu').detach().numpy().copy()
        for i in range(len(token)):
            word = lang.index2word[token[i]]
            sentences[i] = sentences[i] + word + " "
    return sentences


class R2RLang:
    def __init__(self, name: str):
        self.name = name
        # self.word2index = {}
        # self.word2count = {}

        with gzip.open(
            "/home/0/19B30511/av-nav/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train-boseos.json.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
            self.word2index = data["instruction_vocab"]["word2idx_dict"]
            self.index2word = data["instruction_vocab"]["word_list"]
            self.vocab_size = data["instruction_vocab"]["num_vocab"]

        with gzip.open(
            "/home/0/19B30511/av-nav/VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings_gauss.json.gz",
            'rt',
            encoding='utf-8',
        ) as f:
            self.glove_vec = np.array(json.load(f))
