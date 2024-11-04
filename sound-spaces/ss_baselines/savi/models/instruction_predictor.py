import sys
import math
import gzip

import json
import numpy as np
import torch
import torch.nn as nn

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX
from xgenerator.common.lang import tokens2sentences, VIDEO_LLAMA2_TOKENIZER, make_word_histgram


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
        use_bos: bool = False,
        belief_dim: int = 23,
        share_decoder: bool=False,
    ):
        self.vocab_emb_size = vocab_emb_size
        self.emb_size = emb_size
        self.max_instr_len = max_instr_len
        self._pretraining = pretraining
        self.belief_dim = belief_dim
        self.use_bos = use_bos
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

        if self.belief_dim == 23:
            if vocab_emb_size == 50:
                with gzip.open(
                    "/home/4/ud02274/navigation/myss/sound-spaces/data/category_embed/savi_21_categorys.json.gz",
                    'rt',
                    encoding='utf-8',
                ) as f:
                    category_emb_vec = np.array(json.load(f))
            elif vocab_emb_size == 4096:
                with gzip.open(
                    "/home/4/ud02274/navigation/myss/sound-spaces/data/category_embed/savi_21_categorys_4094.json.gz",
                    'rt',
                    encoding='utf-8',
                ) as f:
                    category_emb_vec = np.array(json.load(f))
            else:
                raise NotImplementedError(f"vocab_emb_size: {vocab_emb_size}")
            self.category_emb = nn.Linear(self.belief_dim-2, vocab_emb_size-2)
            self.category_emb.weight.data[...] = torch.from_numpy(category_emb_vec)
            self.category_emb.weight.requires_grad = False
        else:
            self.category_emb = nn.Linear(self.belief_dim-2, vocab_emb_size-2)


        self.word_emb = nn.Linear(vocab_emb_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size,
            dropout=dropout,
        )

        self.share_decoder = share_decoder
        if self.share_decoder:
            self.task_embedding_for_X = nn.Parameter(
                torch.normal(mean=0.0, std=0.1, size=(emb_size,), requires_grad=True)
            )

    def forward(self,
        category: torch.Tensor,
        location: torch.Tensor,
        target: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: torch.Tensor,
        convert_mask: bool = True,
    ):
        if convert_mask:
            memory_key_padding_mask = self.convert_memory_masks(memory_key_padding_mask)

        if target is not None:
            logits = self.teacher_forcing_forward(category, location, target, memory, memory_key_padding_mask)
        else:
            logits = self.student_forcing_forward(category, location, memory, memory_key_padding_mask)
        return logits

    def student_forcing_forward(self, category, location, memory, memory_key_padding_mask):
        batch_size = category.shape[0]
        if self.use_bos:
            raise Exception(f"if you want to use bos, specify tokenizer_type, or you may mistake BOS_IDX.")
        past_tokens = torch.full((1, batch_size), BOS_IDX)
        past_tokens = past_tokens.cuda() if torch.cuda.is_available() else past_tokens
        for i in range(self.max_instr_len-1):
            if i != self.max_instr_len-2:
                with torch.no_grad():
                    if self.use_bos:
                        past_words = self.embed_word_tokens_using_bos(past_tokens)
                    else:
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
                if self.use_bos:
                    past_words = self.embed_word_tokens_using_bos(past_tokens)
                else:
                    past_words = self.embed_word_tokens(past_tokens, category, location)
                decoder_output = self.decoder(
                    tgt=past_words, # (instr_len, batch, embed)
                    memory=memory, # (mem_size, batch, smt_hidden)
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=memory_key_padding_mask,
                )
                logits = self.generator(decoder_output) # (instr_len, batch, vocab_size)
        return logits
    
    def generate(
        self,
        category,
        location,
        memory,
        save_dir_path,
        beam_num,
        top_k,
        top_p,
        temperature,
        tokenizer_type,
    ):
        batch_size = category.shape[0]
        assert batch_size == 1, "The size of batch must be 1 to use generate method."

        if beam_num > 1:
            assert top_k is None and top_p is None and temperature is None
        if top_k is not None or top_p is not None:
            assert beam_num == 1

        if tokenizer_type == "r2r":
            past_tokens = torch.full((1, batch_size), BOS_IDX)
        elif tokenizer_type == "video_llama2":
            past_tokens = torch.full((1, batch_size), 1)
        else:
            raise Exception(f"tokenizer_type: {tokenizer_type}")
        past_tokens = past_tokens.cuda() if torch.cuda.is_available() else past_tokens

        for i in range(self.max_instr_len-1):
            if i == 1:
                memory = memory.repeat(1, beam_num, 1)
                category = category.repeat(beam_num, 1)
                location = location.repeat(beam_num, 1)
                
            # Calculate logits
            with torch.inference_mode():
                if self.use_bos:
                    past_words = self.embed_word_tokens_using_bos(past_tokens)
                else:
                    past_words = self.embed_word_tokens(past_tokens, category, location)
                tgt_mask = torch.triu(torch.full((i+1, i+1), 1), diagonal=1).type(torch.bool)
                tgt_mask = tgt_mask.cuda() if torch.cuda.is_available() else tgt_mask

                decoder_output = self.decoder(
                    tgt=past_words, # (instr_len, batch, embed)
                    memory=memory, # (mem_size(=152), batch, smt_hidden)
                    tgt_mask=tgt_mask,
                    memory_key_padding_mask=None,
                )[-1, :, :]
                logits = self.generator(decoder_output) # (batch, vocab_size)  

            if beam_num > 1: # Beam search              
                probs = nn.functional.softmax(logits, dim=1)

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
                    top_k = min(len(topp_indices), top_k) if top_k is not None else min(len(topp_indices), 20)

                if save_dir_path is not None:
                    make_word_histgram(
                        probs=probs, # (1, vocab_size)
                        top_k=top_k,
                        tokenizer_type=tokenizer_type,
                        save_dir_path=f"{save_dir_path}/hist",
                        filename=f"{i}.png",
                        past_tokens=past_tokens, # (instr_len, batch)
                    )

                probs = probs / torch.sum(probs, dim=1)

                probs = probs.flatten()
                token = torch.multinomial(probs, num_samples=1)
                past_tokens = torch.cat((past_tokens, token.unsqueeze(0)), dim=0)

            # sentence = tokens2sentences(past_tokens[:, 0].view(-1,).view(-1, 1))[0]
            # sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(past_tokens[:, 0].view(-1,).unsqueeze(0), skip_special_tokens=False)[0]
            # print(f"sentence: {sentence}")

        return past_tokens[:, 0].view(-1,)   

    def teacher_forcing_forward(self, category, location, target, memory, memory_key_padding_mask):
        _, instr_len = target.size()
        target = target[:, :-1].permute(1, 0).long() # (instr_len, batch)
        
        tgt_mask = torch.triu(torch.full((instr_len-1, instr_len-1), 1), diagonal=1).type(torch.bool)
        tgt_mask = tgt_mask.to(target.device)
        if self.use_bos:
            past_words = self.embed_word_tokens_using_bos(target)
        else:
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
        if self.share_decoder:
            words = words + self.task_embedding_for_X
        return words
    
    def embed_word_tokens_using_bos(self, word_tokens):
        """
        word_tokens: (instr_len, batch)
        """
        instr_len, batch_size = word_tokens.shape
        word_tokens = word_tokens.reshape(instr_len * batch_size) # (instr_len*batch,)
        embs = self.word_vocab_emb(word_tokens) # (instr_len*batch, 50)
        words = self.word_emb(embs) # (instr_len*batch, embed)
        words = words.view(instr_len, batch_size, self.emb_size) # (instr_len, batch, embed)
        words = self.positional_encoding(words)
        if self.share_decoder:
            words = words + self.task_embedding_for_X
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
