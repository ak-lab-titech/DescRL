import sys
import gzip
import json

import numpy as np
import torch

sys.path.append("/home/4/ud02274/navigation/myss/xgenerator")

from common.load_lmdb import get_all_lmdb_data, EOS_IDX


def calc_confidence(logits):
    """
    logits: (seq_len, 1, word_num)
    """
    sm = torch.nn.Softmax(dim=2)
    probs = sm(logits)
    max_value, max_idx = torch.max(probs, dim=2) # (seq_len, 1), (seq_len, 1)
    
    max_value = max_value.detach().cpu().numpy().reshape(-1,) # (seq_len,)
    max_idx = max_idx.detach().cpu().numpy().reshape(-1,)   # (seq_len,)
    eos_idx_list = np.where(max_idx == EOS_IDX)[0]

    if len(eos_idx_list) > 0:
        eos_idx = eos_idx_list[0]
        confidence = np.mean(max_value[:eos_idx+1])
    else:
        confidence = np.mean(max_value)
    return confidence



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
            "/home/4/ud02274/navigation/my-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train-boseos.json.gz",
            "rt",
            encoding="utf-8",
        ) as f:
            data = json.load(f)
            self.word2index = data["instruction_vocab"]["word2idx_dict"]
            self.index2word = data["instruction_vocab"]["word_list"]
            self.vocab_size = data["instruction_vocab"]["num_vocab"]

        with gzip.open(
            "/home/4/ud02274/navigation/my-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/embeddings_gauss.json.gz",
            'rt',
            encoding='utf-8',
        ) as f:
            self.glove_vec = np.array(json.load(f))
