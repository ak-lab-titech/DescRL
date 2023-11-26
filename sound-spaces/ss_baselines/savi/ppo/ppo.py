#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.append("/home/0/19B30511/av-nav/myss")
from xgenerator.common.load_lmdb import PAD_IDX

EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic,
        use_iprl,
        clip_param,
        ppo_epoch,
        num_mini_batch,
        iprl_loss_coef,
        value_loss_coef,
        entropy_coef,
        direct_map_loss_coef,
        lr=None,
        eps=None,
        max_grad_norm=None,
        use_clipped_value_loss=True,
        use_normalized_advantage=True,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.use_iprl = use_iprl
        self.iprl_loss_coef = iprl_loss_coef
        self.iprl_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)


        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.direct_map_loss_coef = direct_map_loss_coef
        

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr, eps=eps)

        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0
        direct_map_loss_epoch = 0
        iprl_loss_epoch = 0

        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                advantages, self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_direct_map_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    external_memory,
                    external_memory_masks,
                ) = sample

                # Reshape to do in a single forward pass for all steps
                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    _,
                    predict_direct_map,
                    iprl_logits,
                ) = self.actor_critic.evaluate_actions(
                    obs_batch,
                    recurrent_hidden_states_batch,
                    prev_direct_map_batch,
                    prev_actions_batch,
                    masks_batch,
                    actions_batch,
                    external_memory,
                    external_memory_masks,
                )

                ratio = torch.exp(
                    action_log_probs - old_action_log_probs_batch
                )
                surr1 = ratio * adv_targ
                surr2 = (
                    torch.clamp(
                        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
                    )
                    * adv_targ
                )
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_preds_batch + (
                        values - value_preds_batch
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (
                        value_pred_clipped - return_batch
                    ).pow(2)
                    value_loss = (
                        0.5
                        * torch.max(value_losses, value_losses_clipped).mean()
                    )
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()
                
                if predict_direct_map is not None:
                    direct_map_loss = 0.5 * (obs_batch["direct_map"] - predict_direct_map).pow(2).mean()
                else:
                    direct_map_loss = 0
                
                if self.use_iprl:
                    # logits: (instr_len, batch, vocab_size)
                    iprl_targets = obs_batch["generated_instruction"].permute(1, 0)[1:, :].long() # (instr_len, batch)
                    iprl_loss = self.iprl_loss_fn(iprl_logits.reshape(-1, iprl_logits.shape[-1]), iprl_targets.reshape(-1))
                else:
                    iprl_loss = 0

                self.optimizer.zero_grad()

                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + direct_map_loss * self.direct_map_loss_coef
                    + iprl_loss * self.iprl_loss_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                if predict_direct_map is not None:
                    direct_map_loss_epoch += direct_map_loss.item()
                else:
                    direct_map_loss_epoch += 0
                if self.use_iprl:
                    iprl_loss_epoch += iprl_loss.item()
                else:
                    iprl_loss_epoch += 0

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        direct_map_loss_epoch /= num_updates
        iprl_loss_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch, direct_map_loss_epoch, iprl_loss_epoch

    def before_backward(self, loss):
        pass

    def after_backward(self, loss):
        pass

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.actor_critic.parameters(), self.max_grad_norm
        )

    def after_step(self):
        pass
