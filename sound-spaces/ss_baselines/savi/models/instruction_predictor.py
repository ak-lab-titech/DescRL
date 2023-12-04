import sys
import math
import gzip

import json
import numpy as np
import torch
import torch.nn as nn

sys.path.append("/home/0/19B30511/av-nav/myss")
from xgenerator.common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX


class PositionalEncoding(nn.Module):
    def __init__(self,
        emb_size: int,
        dropout: float,
        maxlen: int = 200,
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

    def forward(self, token_embedding: torch.Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


class InstructionPredictor(nn.Module):
    def __init__(
        self,
        num_decoder_layers: int,
        vocab_emb_size: int,
        emb_size: int,
        max_instr_len: int,
        nhead: int,
        vocab_size: int,
        glove: np.ndarray,
        dim_feedforward: int,
        dropout: float,
        pretraining: bool = False,
    ):
        self.vocab_emb_size = vocab_emb_size
        self.emb_size = emb_size
        self.max_instr_len = max_instr_len
        self._pretraining = pretraining
        self.belief_dim = 23
        super(InstructionPredictor, self).__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, vocab_size)

        self.word_vocab_emb = nn.Embedding(vocab_size, vocab_emb_size)
        if glove is not None:
            print('Using GloVe embedding')
            self.word_vocab_emb.weight.data[...] = torch.from_numpy(glove)
            self.word_vocab_emb.weight.requires_grad = False

        with gzip.open(
            "/home/0/19B30511/av-nav/myss/sound-spaces/data/category_embed/savi_21_categorys.json.gz",
            'rt',
            encoding='utf-8',
        ) as f:
            category_emb_vec = np.array(json.load(f))
        self.category_emb = nn.Linear(self.belief_dim-2, vocab_emb_size-2)
        self.category_emb.weight.data[...] = torch.from_numpy(category_emb_vec)
        self.category_emb.weight.requires_grad = False

        self.word_emb = nn.Linear(vocab_emb_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout,
        )

    def forward(self,
        category: torch.Tensor,
        location: torch.Tensor,
        target: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor
    ):
        memory_key_padding_mask = self.convert_memory_masks(memory_key_padding_mask)

        if target is not None:
            logits = self.teacher_forcing_forward(category, location, target, memory, memory_key_padding_mask)
        else:
            logits = self.student_forcing_forward(category, location, memory, memory_key_padding_mask)
        return logits

    def student_forcing_forward(self, category, location, memory, memory_key_padding_mask):
        batch_size = category.shape[0]     
        past_tokens = torch.full((1, batch_size), BOS_IDX)
        past_tokens = past_tokens.cuda() if torch.cuda.is_available() else past_tokens
        for i in range(self.max_instr_len-1):
            if i != self.max_instr_len-2:
                with torch.no_grad():
                    past_words = self.embed_word_tokens(past_tokens, category, location)
                    tgt_mask = torch.triu(torch.full((i+1, i+1), 1), diagonal=1).type(torch.bool)
                    tgt_mask = tgt_mask.cuda() if torch.cuda.is_available() else tgt_mask
                    decoder_output = self.decoder(
                        tgt=past_words, # (instr_len, batch, embed)
                        memory=memory, # (mem_size(=152), batch, smt_hidden)
                        tgt_mask=tgt_mask,
                        memory_key_padding_mask=memory_key_padding_mask,
                    )[-1, :, :]
                    logits = self.generator(decoder_output) # (batch, 2506)
                    _, next_token = logits.max(1)
                    next_token = next_token.view(1, batch_size)
                    past_tokens = torch.cat([past_tokens, next_token], dim=0) # (instr_len, batch)
            else:
                tgt_mask = torch.triu(torch.full((i+1, i+1), 1), diagonal=1).type(torch.bool)
                tgt_mask = tgt_mask.cuda() if torch.cuda.is_available() else tgt_mask
                past_words = self.embed_word_tokens(past_tokens, category, location)
                decoder_output = self.decoder(
                    tgt=past_words, # (instr_len, batch, embed)
                    memory=memory, # (mem_size, batch, smt_hidden)
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.generator(decoder_output) # (instr_len, batch, vocab_size)
        return logits

    def teacher_forcing_forward(self, category, location, target, memory, memory_key_padding_mask):
        _, instr_len = target.size()
        target = target[:, :-1].permute(1, 0).long() # (instr_len, batch)
        
        tgt_mask = torch.triu(torch.full((instr_len-1, instr_len-1), 1), diagonal=1).type(torch.bool)
        tgt_mask = tgt_mask.cuda() if torch.cuda.is_available() else tgt_mask
        past_words = self.embed_word_tokens(target, category, location)
        decoder_output = self.decoder(
            tgt=past_words, # (instr_len, batch, embed)
            memory=memory, # (mem_size, batch, smt_hidden)
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        logits = self.generator(decoder_output) # (instr_len, batch, vocab_size)
        return logits

    
    def embed_word_tokens(self, word_tokens, category, location):
        """
        word_tokens: (instr_len, batch)
        category: (batch, 21)
        location: (batch, 2)
        """
        instr_len, batch_size = word_tokens.shape
        embs = torch.cat([self.category_emb(category), location], dim=1) # (batch, 50)
        embs = embs.view(1, batch_size, self.vocab_emb_size)
        if instr_len > 1:
            word_tokens = word_tokens[1:, :].reshape((instr_len-1)*batch_size) # (instr_len*batch, )
            word_vocab_embs = self.word_vocab_emb(word_tokens) # (instr_len*batch, 50)
            embs = torch.cat(
                [
                    embs,
                    word_vocab_embs.view(instr_len-1, batch_size, self.vocab_emb_size),
                ],
                dim=0,
            ) # (instr_len, batch, embed)
        embs = embs.view(instr_len*batch_size, self.vocab_emb_size)
        words = self.word_emb(embs) # (instr_len*batch, embed)
        words = words.view(instr_len, batch_size, self.emb_size) # (instr_len, batch, embed)
        words = self.positional_encoding(words)
        return words
    
    def convert_memory_masks(self, memory_masks):
        if self._pretraining:
            memory_masks = torch.cat(
                [
                    torch.zeros_like(memory_masks),
                    torch.ones([memory_masks.shape[0], 1], device=memory_masks.device),
                ],
                dim=1,
            )
        else:
            memory_masks = torch.cat(
                [
                    memory_masks,
                    torch.ones([memory_masks.shape[0], 1], device=memory_masks.device),
                ],
                dim=1,
            )
        return (1 - memory_masks) > 0
