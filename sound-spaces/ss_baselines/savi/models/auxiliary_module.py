import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import layer_init


def convert_memory_masks(memory_masks, pretraining):
    if pretraining:
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


class ProgressMonitorPredictor(nn.Module):
    def __init__(
        self,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        pretraining,
    ):
        super().__init__()
        self._pretraining = pretraining

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.linear = nn.Linear(emb_size, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(
        self,
        memory,
        target,
        memory_key_padding_mask,
        convert_mask: bool = True,
    ):
        if convert_mask:
            memory_key_padding_mask = convert_memory_masks(memory_key_padding_mask, self._pretraining)

        decoder_output = self.decoder(
            tgt=target, # (1, batch, smt_hidden)
            memory=memory, # (mem_size(=152), batch, smt_hidden)
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        predicted_progress = self.sigmoid(self.linear(decoder_output))

        return predicted_progress


class NextFramePredictor(nn.Module):
    def __init__(
        self,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        pretraining,
    ):
        super().__init__()
    
        self._pretraining = pretraining

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )

        output_channels = 4 # TODO 適当なので要修正
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(
        self,
        memory,
        action,
        memory_key_padding_mask,
        convert_mask: bool = True,
    ):
        if convert_mask:
            memory_key_padding_mask = convert_memory_masks(memory_key_padding_mask, self._pretraining)

        decoder_output = self.decoder(
            tgt=action,
            memory=memory, # (mem_size(=152), batch, smt_hidden)
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        decoder_output = decoder_output.view(-1, 4, 128, 128) # param適当
        
        predicted_frame = self.conv_transpose(decoder_output)

        return predicted_frame


class NextOracleActionPredictor(nn.Module):
    def __init__(
        self,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        pretraining,
    ):
        super().__init__()
        self._pretraining = pretraining

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=num_decoder_layers,
        )
        self.linear = nn.Linear(emb_size, 4)

    def forward(
        self,
        memory,
        target,
        memory_key_padding_mask,
        convert_mask: bool = True,
    ):
        if convert_mask:
            memory_key_padding_mask = convert_memory_masks(memory_key_padding_mask, self._pretraining)

        decoder_output = self.decoder(
            tgt=target,
            memory=memory, # (mem_size(=152), batch, smt_hidden)
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )

        logits = self.linear(decoder_output)
        return logits
