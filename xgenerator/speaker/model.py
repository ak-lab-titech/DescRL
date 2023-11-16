import sys

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.utils import try_cuda
from common.model import (
    VisualFeatureEncoder,
    VisualImageEncoder,
    ContextOnlySoftDotAttention,
    SoftDotAttention,
)


class SpeakerEncoderLSTM(nn.Module):
    def __init__(self, action_embedding_size, world_embedding_size,
                 hidden_size, dropout_ratio, use_image_feature, bidirectional=False):
        super(SpeakerEncoderLSTM, self).__init__()
        assert not bidirectional, 'Bidirectional is not implemented yet'

        self.action_embedding_size = action_embedding_size
        self.word_embedding_size = world_embedding_size
        self.hidden_size = hidden_size
        self.drop = nn.Dropout(p=dropout_ratio)
        # self.visual_attention_layer = VisualSoftDotAttention(
        #     hidden_size, world_embedding_size
        # )
        if use_image_feature:
            # feature param: 34,816
            self.visual_encoder = VisualFeatureEncoder((2176, 4, 4), world_embedding_size)
        else:
            # Image param: 256*256*4 = 262,144
            self.visual_encoder = VisualImageEncoder((256, 256, 4), world_embedding_size)
        self.lstm = nn.LSTMCell(
            action_embedding_size + world_embedding_size, hidden_size
        )
        self.encoder2decoder = nn.Linear(hidden_size, hidden_size)

    def init_state(self, batch_size):
        ''' Initialize to zero cell states and hidden states.'''
        h0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        c0 = Variable(
            torch.zeros(batch_size, self.hidden_size), requires_grad=False)
        return try_cuda(h0), try_cuda(c0)

    def _forward_one_step(self, h_0, c_0, action_embedding,
                          world_state_embedding):
        # feature, _ = self.visual_attention_layer(h_0, world_state_embedding)
        feature = self.visual_encoder(world_state_embedding) # (b, v_dim)
        concat_input = torch.cat((action_embedding, feature), 1) # (b, v_dim + a_dim)
        drop = self.drop(concat_input) # (b, v_dim + a_dim)
        h_1, c_1 = self.lstm(drop, (h_0, c_0)) # h: (b, h_dim), c: (b, h_dim)
        return h_1, c_1

    def forward(self, batched_action_embeddings, world_state_embeddings, seq_lengths):
        ''' Expects action indices as (batch, seq_len). Also requires a
            list of lengths for dynamic batching. '''
        assert isinstance(batched_action_embeddings, list)
        assert isinstance(world_state_embeddings, list)
        assert len(batched_action_embeddings) == len(world_state_embeddings)
        batch_size = world_state_embeddings[0].shape[0]

        h, c = self.init_state(batch_size) # h: (batch, hidden_size), c: (batch, hidden_size)
        h_list = []
        c_list = []
        for t, (action_embedding, world_state_embedding) in enumerate(
                zip(batched_action_embeddings, world_state_embeddings)):
            h, c = self._forward_one_step(
                h, c, action_embedding, world_state_embedding
            )
            h_list.append(h)
            c_list.append(c)

        decoder_init = nn.Tanh()(self.encoder2decoder(
            torch.stack([h_list[seq_length-1][i] for i, seq_length in enumerate(seq_lengths)])
        )) # (batch, hidden_size)
        c = torch.stack([c_list[seq_length-1][i] for i, seq_length in enumerate(seq_lengths)])

        ctx = torch.stack(h_list, dim=1)  # (batch, seq_len, hidden_size)
        ctx = self.drop(ctx)
        return ctx, decoder_init, c


class SpeakerDecoderLSTM(nn.Module):
    def __init__(self, vocab_size, vocab_embedding_size, hidden_size,
                 dropout_ratio, glove=None, use_input_att_feed=False):
        super(SpeakerDecoderLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.vocab_embedding_size = vocab_embedding_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, vocab_embedding_size)
        self.use_glove = glove is not None
        if self.use_glove:
            print('Using GloVe embedding')
            self.embedding.weight.data[...] = torch.from_numpy(glove)
            self.embedding.weight.requires_grad = False
        self.drop = nn.Dropout(p=dropout_ratio)
        self.use_input_att_feed = use_input_att_feed
        if self.use_input_att_feed:
            print('using input attention feed in SpeakerDecoderLSTM')
            self.lstm = nn.LSTMCell(
                vocab_embedding_size + hidden_size, hidden_size)
            self.attention_layer = ContextOnlySoftDotAttention(hidden_size)
            self.output_l1 = nn.Linear(hidden_size*2, hidden_size)
            self.tanh = nn.Tanh()
        else:
            self.lstm = nn.LSTMCell(vocab_embedding_size, hidden_size)
            self.attention_layer = SoftDotAttention(hidden_size)
        self.decoder2action = nn.Linear(hidden_size, vocab_size)
         

    def forward(self, previous_word, h_0, c_0, ctx, ctx_mask=None):
        ''' Takes a single step in the decoder LSTM (allowing sampling).

        action: batch x 1
        feature: batch x feature_size
        h_0: batch x hidden_size
        c_0: batch x hidden_size
        ctx: batch x seq_len x dim
        ctx_mask: batch x seq_len - indices to be masked
        '''
        word_embeds = self.embedding(previous_word)
        word_embeds = word_embeds.squeeze()  # (batch, embedding_size)
        if not self.use_glove:
            word_embeds_drop = self.drop(word_embeds)
        else:
            word_embeds_drop = word_embeds

        if self.use_input_att_feed:
            h_tilde, alpha = self.attention_layer(
                self.drop(h_0), ctx, ctx_mask)
            concat_input = torch.cat((word_embeds_drop, self.drop(h_tilde)), 1)
            h_1, c_1 = self.lstm(concat_input, (h_0, c_0))
            x = torch.cat((h_1, h_tilde), 1)
            x = self.drop(x)
            x = self.output_l1(x)
            x = self.tanh(x)
            logit = self.decoder2action(x)
        else:
            # h_1: (batch_size, hidden_size), c_1: (batch_size, hidden_size)
            h_1, c_1 = self.lstm(word_embeds_drop, (h_0, c_0))
            h_1_drop = self.drop(h_1)
            # h_tilde: (batch_size, hidden_size), alpha: (batch_size, seq_len)
            h_tilde, alpha = self.attention_layer(h_1_drop, ctx, ctx_mask)
            logit = self.decoder2action(h_tilde) # (batch_size, vocab_size)
        return h_1, c_1, alpha, logit

