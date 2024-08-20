import sys
import math
from typing import Optional
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Transformer

sys.path.append("/home/4/ud02274/navigation/myss/xgenerator")
from common.load_lmdb import BOS_IDX
from common.model import (
    VisualFeatureEncoder,
    VisualImageEncoder,
)
from xgenerator.common.lang import tokens2sentences, make_word_histgram


class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout: float,
        maxlen: int = 200
    ):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        vocab_emb_size: int,
        emb_size: int,
        nhead: int,
        use_image_feature: int,
        vocab_size: int,
        use_semantic: bool,
        glove: np.ndarray = None,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.generator = nn.Linear(emb_size, vocab_size)
        if use_image_feature:
            # feature param: 2176*4*4 = 34,816
            print("Visual image feature shape: (2176, 4, 4)")
            self.visual_emb = VisualFeatureEncoder((2176, 4, 4), emb_size-4)
        else:
            # Image param: 256*256*4 = 262,144
            #              128*128*4 = 65,536
            # self.visual_emb = VisualImageEncoder((256, 256, 4), emb_size-4)
            if use_semantic:
                print("Visual image shape: (128, 128, 7)")
                self.visual_emb = VisualImageEncoder((128, 128, 7), emb_size-4)
            else:
                print("Visual image shape: (128, 128, 4)")
                self.visual_emb = VisualImageEncoder((128, 128, 4), emb_size-4)
        self.word_vocab_emb = nn.Embedding(vocab_size, vocab_emb_size)
        if glove is not None:
            print('Using GloVe embedding')
            self.word_vocab_emb.weight.data[...] = torch.from_numpy(glove)
            self.word_vocab_emb.weight.requires_grad = False
        self.word_emb = nn.Linear(vocab_emb_size, emb_size)
        
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout,
        )

    def forward(self,
        src_image: Tensor,
        src_action: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        memory_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_image_shape = src_image.shape
        src_action_shape = src_action.shape
        src_image = src_image.view(src_image_shape[0] * src_image_shape[1], src_image_shape[2], src_image_shape[3], src_image_shape[4])
        src_action = src_action.view(src_action_shape[0] * src_action_shape[1], src_action_shape[2])
        src_emb = torch.cat((self.visual_emb(src_image), src_action), 1) # (max_l*batch, embed)
        _, emb_size = src_emb.shape
        src_emb = src_emb.view(src_image_shape[0], src_image_shape[1], emb_size) # (max_l, batch, embed)
        src_emb = self.positional_encoding(src_emb)
        trg_shape = trg.shape
        trg = trg.reshape(-1) # (199*batch, )
        tgt_emb = self.word_vocab_emb(trg) # (199*batch, 50)
        tgt_emb = self.word_emb(tgt_emb) # (199*batch, 512)
        tgt_emb = tgt_emb.view(trg_shape[0], trg_shape[1], emb_size)
        tgt_emb = self.positional_encoding(tgt_emb)
        outs = self.transformer(
            src=src_emb,
            tgt=tgt_emb,
            src_mask=src_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_mask=memory_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)
    
    def generate(
        self,
        image_seqs: Tensor, # (l, b, h, w, c)
        action_seqs: Tensor, # (l, b, 4)
        max_instr_len: int,
        save_dir_path: str,
        beam_num: int = 1,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
    ):
        batch_size = np.shape(image_seqs)[1]
        assert batch_size == 1, "batch_size must be 1."

        if beam_num > 1:
            assert top_k is None and top_p is None and temperature is None
        if top_k is not None or top_p is not None:
            assert beam_num == 1

        with torch.inference_mode():
            memory = self.encode(
                src_image=image_seqs,
                src_action=action_seqs,
                src_mask=None,
                src_padding_mask=None,
            )
        past_tokens = torch.full((1, batch_size), BOS_IDX)
        past_tokens = past_tokens.cuda() if torch.cuda.is_available() else past_tokens

        for i in range(max_instr_len-1):

            with torch.inference_mode():
                tgt_mask = torch.triu(torch.full((i+1, i+1), 1), diagonal=1).type(torch.bool)
                tgt_mask = tgt_mask.cuda() if torch.cuda.is_available() else tgt_mask

                if i == 1:
                    memory = memory.repeat(1, beam_num, 1)

                logits = self.decode(
                    trg=past_tokens,
                    memory=memory,
                    tgt_mask=tgt_mask,
                    memory_mask=None,
                    tgt_padding_mask=None,
                    memory_key_padding_mask=None,
                )[-1, :, :] # (batch, vocab_size)
            
            if beam_num > 1: # Beam search
                probs = nn.functional.softmax(logits, dim=1) # (batch, vocab_size)

                top_indices = np.argsort(probs.flatten().to('cpu').detach().numpy().copy())[::-1][:beam_num]
                top_indices = np.unravel_index(top_indices, probs.shape)

                beams = []
                for j in range(beam_num):
                    beam_j = past_tokens[:, top_indices[0][j]].view(-1,).to('cpu').detach().numpy().copy().tolist() + [top_indices[1][j]]
                    beams.append(beam_j)

                past_tokens = torch.from_numpy(np.array(beams)).permute(1, 0).cuda()
            else:
                assert logits.shape[0] == 1, "The size of batch must be 1."

                probs = nn.functional.softmax(logits / temperature, dim=1)

                if top_k is not None:
                    topk_values, topk_indices = torch.topk(probs, top_k, dim=1)
                    probs = torch.zeros_like(probs)
                    probs[0, topk_indices] = topk_values
                    
                if top_p is not None:
                    sorted_values, sorted_indices = torch.sort(probs, descending=True)
                    cumulative_values = torch.cumsum(sorted_values, dim=1)
                    top_p_mask = cumulative_values <= top_p
                    top_p_mask[:, 0] = True # 少なくとも最大値は入れておく

                    topp_values = sorted_values[top_p_mask]
                    topp_indices = sorted_indices[top_p_mask]
                    probs = torch.zeros_like(probs)
                    probs[0, topp_indices] = topp_values

                    top_k = min(len(topp_indices), top_k) if top_k is not None else min(len(top_indices), 20)

                make_word_histgram(
                    probs=probs, # (1, vocab_size)
                    top_k=top_k,
                    tokenizer_type="r2r",
                    save_dir_path=f"{save_dir_path}/hist",
                    filename=f"{i}.png",
                    past_tokens=past_tokens, # (instr_len, batch)
                )
                
                probs = probs / torch.sum(probs, dim=1)
                probs = probs.flatten()
                token = torch.multinomial(probs, num_samples=1)
                past_tokens = torch.cat((past_tokens, token.unsqueeze(0)), dim=0)
            
            # sentence = tokens2sentences(past_tokens[:, 0].view(-1,).view(-1, 1))[0]
            # print(f"sentence: {sentence}")

        return past_tokens[:, 0].view(-1,)        

    def encode(
        self,
        src_image: Tensor,
        src_action: Tensor,
        src_mask: Tensor,
        src_padding_mask: Tensor,
    ):
        src_image_shape = src_image.shape
        src_action_shape = src_action.shape
        src_image = src_image.view(src_image_shape[0] * src_image_shape[1], src_image_shape[2], src_image_shape[3], src_image_shape[4])
        src_action = src_action.view(src_action_shape[0] * src_action_shape[1], src_action_shape[2])
        src_emb = torch.cat((self.visual_emb(src_image), src_action), 1) # (max_l*batch, embed)
        _, emb_size = src_emb.shape
        src_emb = src_emb.view(src_image_shape[0], src_image_shape[1], emb_size) # (max_l, batch, embed)
        src_emb = self.positional_encoding(src_emb)

        memory = self.transformer.encoder(
            src_emb,
            mask=src_mask,
            src_key_padding_mask=src_padding_mask,
        )
        return memory

    def decode(
        self,
        trg: Tensor,
        memory: Tensor,
        tgt_mask: Tensor,
        memory_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        trg_shape = trg.shape
        trg = trg.reshape(-1) # (199*batch, )
        tgt_emb = self.word_vocab_emb(trg) # (199*batch, 50)
        tgt_emb = self.word_emb(tgt_emb) # (199*batch, 512)
        _, emb_size = tgt_emb.shape
        tgt_emb = tgt_emb.view(trg_shape[0], trg_shape[1], emb_size)
        tgt_emb = self.positional_encoding(tgt_emb)
        
        outs = self.transformer.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.generator(outs)
