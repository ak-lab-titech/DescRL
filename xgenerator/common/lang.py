import sys
import gzip
import json
from typing import Dict, Tuple

import numpy as np
import torch
from videollama2.model.builder import load_pretrained_model
from videollama2.mm_utils import get_model_name_from_path

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



def tokens2sentences(tokens, lang=None):
    """
    tokens' shape must be (sentence_len, batch).
    """
    if lang is None:
        lang = R2RLang("r2r")
    sentences = ["" for _ in range(len(tokens[0]))]
    for token in tokens:
        token = token.to('cpu').detach().numpy().copy()
        for i in range(len(token)):
            word = lang.index2word[token[i]]
            sentences[i] = sentences[i] + word + " "
    return sentences


def sentence2token(sentence: str):
    """
    sentence: str
    token: list
    """
    lang = R2RLang("r2r")
    sentence = sentence.split(" ")
    if sentence[-1] == "":
        sentence = sentence[:-1]
    token = []
    for word in sentence:
        word = word.lower()
        if word == "":
            continue
        elif word[-1] == ".":
            if len(word) > 1:
                if word[:-1] in lang.word2index.keys():
                    token.append(lang.word2index[word[:-1]])
                else:
                    token.append(lang.word2index["<unk>"])
            token.append(lang.word2index["."])
        else:
            if word in lang.word2index.keys():
                token.append(lang.word2index[word])
            else:
                token.append(lang.word2index["<unk>"])
    return token


class R2RLang:
    def __init__(self, name: str = "r2r"):
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
        
        self.word_embed_size = self.glove_vec.shape[1]


VIDEO_LLAMA2_TOKENIZER, VIDEO_LLAMA2_MODEL, _, _ = load_pretrained_model(
    "DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base",
    None,
    get_model_name_from_path("DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base"),
)
VIDEO_LLAMA2_VOCAB = VIDEO_LLAMA2_TOKENIZER.get_vocab()
class VideoLLaMA2Lang:
    def __init__(self, name: str = "videollama2"):
        self.name = name

        self.word2index = VIDEO_LLAMA2_VOCAB
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.vocab_size = 32000
        self.glove_vec = VIDEO_LLAMA2_MODEL.get_input_embeddings().weight.cpu().detach().numpy().copy()
        self.word_embed_size = self.glove_vec.shape[1]
