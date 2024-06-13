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


class ResizeSpectrogramModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return nn.functional.interpolate(x, size=(65, 26), mode='bilinear')


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
        self.action_encoder = nn.Linear(4, emb_size)

        output_channels = 4
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
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
            tgt=self.action_encoder(action),
            memory=memory, # (mem_size(=152), batch, smt_hidden)
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoder_output = decoder_output.view(-1, 4, 8, 8)
        
        predicted_frame = self.conv_transpose(decoder_output)
        predicted_frame = predicted_frame.permute(0, 2, 3, 1)

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


class NextSpectrogramPredictor(nn.Module):
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
        self.action_encoder = nn.Linear(4, emb_size)

        output_channels = 2

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(output_channels, output_channels, kernel_size=(3, 4), stride=(2, 2), padding=(1, 1), output_padding=(1, 0)),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1),
            ResizeSpectrogramModel(),
            nn.Sigmoid(),
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
            tgt=self.action_encoder(action),
            memory=memory, # (mem_size(=152), batch, smt_hidden)
            tgt_mask=None,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        decoder_output = decoder_output.view(-1, 2, 16, 8)
        
        predicted_spectrogram = self.conv_transpose(decoder_output)
        predicted_spectrogram = predicted_spectrogram.permute(0, 2, 3, 1)

        return predicted_spectrogram


class SemanticPredictor(nn.Module):
    def __init__(
        self,
        num_decoder_layers,
        emb_size,
        nhead,
        dim_feedforward,
        pretraining,
    ):
        super().__init__()
        raise NotImplementedError() # TODO


class AudioLocationPredictor(nn.Module):
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
        self.linear = nn.Linear(emb_size, 2)
    
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

        predicted_location = self.linear(decoder_output)
        return predicted_location


class AudioCategoryPredictor(nn.Module):
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
        category_num = 21
        self.linear = nn.Linear(emb_size, category_num)
    
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
