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
    ):
        super().__init__()

        self.action_num = action_num

        self.mlp = nn.Sequential(
            nn.Linear(input_size, input_size * 4),
            nn.ReLU(True),
            nn.Linear(input_size * 4, input_size * 2),
            nn.ReLU(True),
            nn.Linear(input_size * 2, input_size),
            nn.ReLU(True),
            nn.Linear(input_size, output_size),
        )

        layer_init(self.mlp)

    def forward(self, prev_direct_map, audio_encoder_output, prev_actions):
        inputs = []
        
        # f = open("debug.txt", "a")
        # f.write(f"audio_encoder_output.shape: {audio_encoder_output.shape}\n")
        # f.close()
        inputs.append(torch.clone(audio_encoder_output))


        prev_actions_onehot = torch.nn.functional.one_hot(prev_actions, num_classes=self.action_num)
        prev_actions_onehot = torch.reshape(prev_actions_onehot, (prev_actions.shape[0], -1))

        # f = open("debug.txt", "a")        
        # f.write(f"prev_actions: {prev_actions}\n")
        # f.write(f"prev_actions.shape: {prev_actions.shape}\n")
        # f.write(f"type of prev_ations: {type(prev_actions)}\n")
        # f.write(f"prev_actions_onehot: {prev_actions_onehot}\n")
        # f.write(f"prev_actions_onehot.shape: {prev_actions_onehot.shape}\n")
        # f.write(f"type of prev_ations_onehot: {type(prev_actions_onehot)}\n")
        # f.close()

        inputs.append(prev_actions_onehot)

        f = open("debug.txt", "a")
        f.write(f"-------------- prev_direct_map in DirectMapEncoder forward ---------------\nhead:\n{prev_direct_map[:5]}\ntail:\n{prev_direct_map[-5:]}\n")
        f.close()
        inputs.append(prev_direct_map)

        inputs = torch.cat(inputs, dim=1)

        # f = open("debug.txt", "a")
        # f.write(f"inputs.shape: {inputs.shape}\n")
        # f.close()

        return self.mlp(inputs)


