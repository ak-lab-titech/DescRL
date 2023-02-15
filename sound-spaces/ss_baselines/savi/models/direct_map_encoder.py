from typing import List

import numpy as np
import torch
import torch.nn as nn

from ss_baselines.common.utils import Flatten
from ss_baselines.av_nav.models.visual_cnn import layer_init


class DirectMapEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        action_num: int,
        use_conv1d: bool = True,
    ):
        super().__init__()

        self.action_num = action_num
        self.direct_map_size = output_size
        self.use_conv1d = use_conv1d
        
        if self.use_conv1d:
            mlp_input_size = input_size - output_size + 32 * output_size

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
            self.mlp = nn.Sequential(
                nn.Linear(mlp_input_size, mlp_input_size * 2),
                nn.ReLU(True),
                nn.Linear(mlp_input_size * 2, mlp_input_size * 2),
                nn.ReLU(True),
                nn.Linear(mlp_input_size * 2, mlp_input_size),
                nn.ReLU(True),
                nn.Linear(mlp_input_size, output_size),
                nn.Sigmoid(),
            )
            layer_init(self.mlp)
            layer_init(self.conv_1d)
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_size, input_size * 4),
                nn.ReLU(True),
                nn.Linear(input_size * 4, input_size * 2),
                nn.ReLU(True),
                nn.Linear(input_size * 2, input_size),
                nn.ReLU(True),
                nn.Linear(input_size, output_size),
                nn.Sigmoid(),
            )
            layer_init(self.mlp)

    def forward(self, prev_direct_map, audio_encoder_output, visual_encoder_output, prev_actions):
        inputs = []

        if visual_encoder_output is not None:
            inputs.append(torch.clone(visual_encoder_output))
        
        inputs.append(torch.clone(audio_encoder_output))

        # prev_actions_onehot = torch.nn.functional.one_hot(prev_actions, num_classes=self.action_num)
        # prev_actions_onehot = torch.reshape(prev_actions_onehot, (prev_actions.shape[0], -1))
        # inputs.append(prev_actions_onehot)
        inputs.append(prev_actions)

        if self.use_conv1d:
            direct_map_latent = self.conv_1d(
                torch.reshape(
                    prev_direct_map,
                    (-1, self.direct_map_size, 1),
                ).permute(0, 2, 1)
            )
            inputs.append(direct_map_latent)
        else:
            inputs.append(prev_direct_map)

        inputs = torch.cat(inputs, dim=1)

        return self.mlp(inputs)


