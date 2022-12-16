from typing import List

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.av_nav.models.visual_cnn import layer_init
from ss_baselines.common.utils import Flatten


class GRUDirectMapEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        action_num: int,
        num_layers: int = 1,
    ):
        super().__init__()

        self.action_num = action_num
        self._num_recurrent_layers = num_layers
        self.direct_map_size = output_size


        self.conv_1d = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding_mode="circular", padding="same"),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 3, padding_mode="circular", padding="same"),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 3, padding_mode="circular", padding="same"),
            nn.ReLU(True),
            nn.Conv1d(32, 32, 3, padding_mode="circular", padding="same"),
            Flatten(),
        )

        rnn_input_size = input_size - output_size + 32 * output_size
        self.rnn = getattr(nn, "GRU")(
            input_size=rnn_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
        )

        mlp_input_size = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size),
            nn.ReLU(True),
            nn.Linear(mlp_input_size, output_size),
            nn.Sigmoid(),
        )

        layer_init(self.conv_1d)
        self.mlp_layer_init()
        self.rnn_layer_init()
    
    def mlp_layer_init(self):
        for layer in self.mlp:
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)
    
    def rnn_layer_init(self):
        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)
    
    @property
    def num_recurrent_layers(self):
        return self._num_recurrent_layers
    
    def _mask_hidden(self, hidden_states, masks):
        if isinstance(hidden_states, tuple):
            hidden_states = tuple(v * masks for v in hidden_states)
        else:
            hidden_states = masks * hidden_states
        return hidden_states
    
    def single_forward(self, x, hidden_states, masks):
        x, hidden_states = self.rnn(
            x.unsqueeze(0),
            self._mask_hidden(hidden_states, masks.unsqueeze(0)),
        )
        x = x.squeeze(0)
        return x, hidden_states

    def seq_forward(self, x, hidden_states, masks):
        # x is a (T, N, -1) tensor flattened to (T * N, -1)
        n = hidden_states.size(1)
        t = int(x.size(0) / n)

        # unflatten
        x = x.view(t, n, x.size(1))
        masks = masks.view(t, n)

        # steps in sequence which have zero for any agent. Assume t=0 has
        # a zero in it.
        has_zeros = (masks[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()

        # +1 to correct the masks[1:]
        if has_zeros.dim() == 0:
            has_zeros = [has_zeros.item() + 1]  # handle scalar
        else:
            has_zeros = (has_zeros + 1).numpy().tolist()

        # add t=0 and t=T to the list
        has_zeros = [0] + has_zeros + [t]

        outputs = []
        for i in range(len(has_zeros) - 1):
            # process steps that don't have any zeros in masks together
            start_idx = has_zeros[i]
            end_idx = has_zeros[i + 1]

            rnn_scores, hidden_states = self.rnn(
                x[start_idx:end_idx],
                self._mask_hidden(
                    hidden_states, masks[start_idx].view(1, -1, 1)
                ),
            )

            outputs.append(rnn_scores)

        # x is a (T, N, -1) tensor
        x = torch.cat(outputs, dim=0)
        x = x.view(t * n, -1)  # flatten
        return x, hidden_states

    def forward(
        self,
        prev_direct_map,
        audio_encoder_output,
        visual_encoder_output,
        prev_actions,
        hidden_states,
        masks,
    ):
        prev_direct_map_cnn_feature = self.conv_1d(
            torch.reshape(
                prev_direct_map,
                (-1, self.direct_map_size, 1),
            ).permute(0, 2, 1)
        )

        x = self.make_inputs(prev_direct_map_cnn_feature, audio_encoder_output, visual_encoder_output, prev_actions)

        if x.size(0) == hidden_states.size(1):
            x, hidden_states = self.single_forward(x, hidden_states, masks)
        else:
            x, hidden_states = self.seq_forward(x, hidden_states, masks)
        
        return self.mlp(x), hidden_states

    def make_inputs(self, prev_direct_map_cnn_feature, audio_encoder_output, visual_encoder_output, prev_actions):
        """
        inputs = (prev_direct_map, (visual,) audio, prev_action_onehot)
        """
        inputs = []

        inputs.append(torch.clone(prev_direct_map_cnn_feature))

        if visual_encoder_output is not None:
            inputs.append(torch.clone(visual_encoder_output))
        
        inputs.append(torch.clone(audio_encoder_output))

        prev_actions_onehot = torch.nn.functional.one_hot(prev_actions, num_classes=self.action_num)
        prev_actions_onehot = torch.reshape(prev_actions_onehot, (prev_actions.shape[0], -1))
        inputs.append(prev_actions_onehot)

        inputs = torch.cat(inputs, dim=1)

        return inputs