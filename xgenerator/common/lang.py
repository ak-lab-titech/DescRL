import sys
import gzip
import json
from typing import Dict, Tuple
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
import torch
from videollama2.model.builder import load_pretrained_model
from videollama2.mm_utils import get_model_name_from_path
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, AutoTokenizer

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


def make_word_histgram(
    probs: torch.Tensor, # (1, vocab_size)
    top_k: int,
    tokenizer_type: str,
    save_dir_path: str,
    filename: str,
    past_tokens: torch.Tensor, # (instr_len, batch)
):
    assert probs.shape[0] == 1
    os.makedirs(f"{save_dir_path}", exist_ok=True)

    sorted_prob, sorted_indices = torch.sort(probs, descending=True)

    if tokenizer_type == "r2r":
        topk_words = tokens2sentences(sorted_indices)[:top_k]
        topk_words = [word.split(" ")[0] for word in topk_words]

        past_sentence = tokens2sentences(past_tokens[:, 0].view(-1,).view(-1, 1))[0]
    elif tokenizer_type == "video_llama2":
        topk_words = VIDEO_LLAMA2_TOKENIZER.batch_decode(sorted_indices.permute(1, 0), skip_special_tokens=False)[:top_k]
        past_sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(past_tokens[:, 0].view(-1,).unsqueeze(0), skip_special_tokens=False)[0]
    else:
        raise Exception(f"tokenizer_type: {tokenizer_type}")
    
    topk_values = sorted_prob.flatten().to('cpu').detach().numpy().copy()[:top_k]
    plt.figure()
    plt.bar(
        range(len(topk_values)),
        topk_values,
        tick_label=topk_words,
    )
    plt.xlabel("Word")
    plt.ylabel("Probability")
    plt.title(f"Input: {past_sentence}")
    plt.savefig(f"{save_dir_path}/{filename}")


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

rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
model_path = "DAMO-NLP-SG/VideoLLaMA2-7B-Base"
world_size = torch.cuda.device_count()
gpu_id = rank % world_size
print(f"GPU ID in lang.py: {gpu_id}")
VIDEO_LLAMA2_TOKENIZER, VIDEO_LLAMA2_MODEL, _, _ = load_pretrained_model(
    model_path,
    None,
    get_model_name_from_path(model_path),
    device=torch.device('cuda', gpu_id),
)
VIDEO_LLAMA2_VOCAB = VIDEO_LLAMA2_TOKENIZER.get_vocab()
VIDEO_LLAMA2_EMBS = VIDEO_LLAMA2_MODEL.get_input_embeddings().weight.cpu().detach().numpy().copy()
del VIDEO_LLAMA2_MODEL
gc.collect()
torch.cuda.empty_cache()

# VIDEO_LLAMA2_VOCAB = None
# VIDEO_LLAMA2_TOKENIZER = None
# VIDEO_LLAMA2_EMBS = None
# VIDOEL_LLAMA2_MODEL = None

class VideoLLaMA2Lang:
    def __init__(self, name: str = "videollama2"):
        self.name = name

        self.word2index = VIDEO_LLAMA2_VOCAB
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.vocab_size = 32000
        self.glove_vec = VIDEO_LLAMA2_EMBS
        self.word_embed_size = self.glove_vec.shape[1]


model_path = "Qwen/Qwen2.5-VL-7B-Instruct"
# rank = int(os.environ["LOCAL_RANK"])
rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
world_size = torch.cuda.device_count()
n_proc = int(os.environ["NP"])
gpu_id = rank % world_size
print(f"GPU ID in lang.py: {gpu_id}")
QWEN_25_VL_MODEL = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", # TODO installにめちゃ時間かかる...
    device_map=torch.device('cuda', gpu_id),
)
QWEN_25_VL_PROCESSOR = AutoProcessor.from_pretrained(model_path)
QWEN_25_VL_TOKENIZER = AutoTokenizer.from_pretrained(model_path)
QWEN_25_VL_VOCAB = QWEN_25_VL_TOKENIZER.get_vocab()
QWEN_25_VL_EMBS = QWEN_25_VL_MODEL.get_input_embeddings().float().weight.cpu().detach().numpy().copy()
del QWEN_25_VL_MODEL
gc.collect()
torch.cuda.empty_cache()
class Qwen25VLLang:
    def __init__(self, name: str = "qwen25vl"):
        self.name = name

        self.word2index = QWEN_25_VL_VOCAB
        self.index2word = {v: k for k, v in self.word2index.items()}
        self.vocab_size = np.shape(QWEN_25_VL_EMBS)[0]
        self.glove_vec = QWEN_25_VL_EMBS
        self.word_embed_size = self.glove_vec.shape[1]
