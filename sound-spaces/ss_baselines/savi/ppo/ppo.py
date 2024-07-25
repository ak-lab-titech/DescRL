#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from habitat import logger
sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.load_lmdb import PAD_IDX
from xgenerator.common.lang import tokens2sentences, R2RLang
from ss_baselines.savi.iprl_pretraining.common.videollama2_kd_loss import VideoLLaMA2KDLoss, R2RTokenizerVideoLLaMA2KDLoss
from xgenerator.common.lang import tokens2sentences, sentence2token, R2RLang, VIDEO_LLAMA2_TOKENIZER
from xgenerator.common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX

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
        xgenerator_type=None,
        xgenerator_tokenizer_type=None,
    ):

        super().__init__()

        self.actor_critic = actor_critic

        self.use_iprl = use_iprl
        self.iprl_loss_coef = iprl_loss_coef

        self.xgenerator_type = xgenerator_type
        self.xgenerator_tokenizer_type = xgenerator_tokenizer_type
        logger.info(f"xgenerator_type: {self.xgenerator_type}, tokenizer_type: {self.xgenerator_tokenizer_type}")
        if self.xgenerator_type == "cnn_tf":
            self.iprl_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        elif self.xgenerator_type == "video_llama2" and self.xgenerator_tokenizer_type == "video_llama2":
            self.iprl_loss_fn = VideoLLaMA2KDLoss(visual_feature_coef=0.0, logits_coef=1.0)
        elif self.xgenerator_type == "video_llama2" and self.xgenerator_tokenizer_type == "r2r":
            self.iprl_loss_fn = R2RTokenizerVideoLLaMA2KDLoss(visual_feature_coef=0.0, logits_coef=1.0)
        else:
            raise Exception(f"xgenerator_type: {self.xgenerator_type}, tokenizer_type: {self.xgenerator_tokenizer_type}")
        self.predict_action_loss = torch.nn.CrossEntropyLoss()
        self.predict_progress_loss = torch.nn.MSELoss()
        self.predict_next_frame_loss = torch.nn.MSELoss()
        self.predict_next_spectrogram_loss = torch.nn.MSELoss()
        self.predict_audio_location_loss = torch.nn.MSELoss()
        self.predict_audio_category_loss = torch.nn.CrossEntropyLoss()

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
        self.update_cnt = 0

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
                    aux_infos,
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
                    # print(f"aux_infos: {aux_infos}")
                    iprl_loss = 0
                    if aux_infos["logits"] is not None:
                        iprl_logits = aux_infos["logits"] # logits: (instr_len, batch, vocab_size)
                        if self.xgenerator_type == "cnn_tf":
                            if "generated_instruction" in obs_batch.keys():
                                iprl_targets = obs_batch["generated_instruction"].permute(1, 0)[1:, :].long() # (instr_len, batch)
                            elif "habitat_sim_generated_instruction" in obs_batch.keys():
                                iprl_targets = obs_batch["habitat_sim_generated_instruction"].permute(1, 0)[1:, :].long() # (instr_len, batch)
                            else:
                                raise Exception("use_iprl is True, but there is no generated instruction.")
                            iprl_loss += self.iprl_loss_fn(iprl_logits.view(-1, iprl_logits.shape[-1]), iprl_targets.reshape(-1))
                            if int(os.environ["LOCAL_RANK"]) == 0 and e == 0 and self.update_cnt % 5 == 0:
                                lang = R2RLang("r2r")
                                _, iprl_tokens = iprl_logits.max(2)
                                pred_sentence = tokens2sentences(iprl_tokens, lang)
                                true_sentence = tokens2sentences(iprl_targets, lang)
                                logger.info(f"Pred -1: {pred_sentence[-1]}")
                                logger.info(f"True -1: {true_sentence[-1]}")
                        elif self.xgenerator_type == "video_llama2":
                            if "generated_instruction" in obs_batch.keys():
                                if self.xgenerator_tokenizer_type == "r2r":
                                    teacher_targets = aux_infos["eprl_target"] # (batch, instr_len)
                                    loss, _ = self.iprl_loss_fn(
                                        teacher_label=teacher_targets.permute(1, 0)[:-1], # (instr_len-1, batch)
                                        student_logits=iprl_logits,
                                        teacher_visual_features=None,
                                        student_visual_features=None,
                                        path_mask=None,
                                    )
                                    iprl_loss += loss
                                    if int(os.environ["LOCAL_RANK"]) == 0 and e == 0 and self.update_cnt % 5 == 0:
                                        lang = R2RLang()
                                        _, iprl_tokens = iprl_logits.max(2)
                                        pred_sentence = tokens2sentences(iprl_tokens, lang)[0]
                                        true_sentence = tokens2sentences(teacher_targets.permute(1, 0)[:-1], lang)[0]
                                        logger.info(f"Pred -1: {pred_sentence}")
                                        logger.info(f"True -1: {true_sentence}")
                                elif self.xgenerator_tokenizer_type == "video_llama2":
                                    teacher_logits = obs_batch["generated_instruction"]
                                    teacher_logits_mask = (teacher_logits == 0).all(dim=2).long()

                                    loss, _ = self.iprl_loss_fn(
                                        teacher_logits=teacher_logits.permute(1, 0, 2)[:-1], # (instr_len-1, batch, vocab_size)
                                        student_logits=iprl_logits, # (instr_len-1, batch, vocab_size)
                                        teacher_logits_mask=teacher_logits_mask[:, :-1], # (batch, instr_len-1)
                                        teacher_visual_features=None, # (seq_len, batch, dim)
                                        student_visual_features=None, # (seq_len, batch, dim)
                                        path_mask=None,
                                    )
                                    iprl_loss += loss
                                    if int(os.environ["LOCAL_RANK"]) == 0 and e == 0 and self.update_cnt % 5 == 0:
                                        _, iprl_tokens = iprl_logits.max(2)
                                        _, target_tokens = teacher_logits.permute(1, 0, 2)[:-1].max(2)
                                        pred_sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(iprl_tokens.permute(1, 0), skip_special_tokens=True)[0]
                                        true_sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(target_tokens.permute(1, 0), skip_special_tokens=True)[0]
                                        logger.info(f"Pred -1: {pred_sentence}")
                                        logger.info(f"True -1: {true_sentence}")
                                else:
                                    raise Exception(f"xgenerator_tokenizer_type: {self.xgenerator_tokenizer_type}")
                            else:
                                raise NotImplementedError()
                        else:
                            raise Exception(f"xgenerator_type: {self.xgenerator_type}")

                    if aux_infos["predicted_progress"] is not None:
                        predicted_progress = aux_infos["predicted_progress"]
                        gt_progress = obs_batch["progress_monitor"]
                        iprl_loss += self.predict_progress_loss(predicted_progress.view(-1, predicted_progress.shape[-1]), gt_progress.reshape(-1))
                    
                    if aux_infos["predicted_action"] is not None:
                        predicted_action = aux_infos["predicted_action"]
                        gt_action = obs_batch["next_optimal_action"]
                        iprl_loss += self.predict_action_loss(predicted_action.view(-1, predicted_action.shape[-1]), gt_action.reshape(-1).long())
                    
                    if aux_infos["predicted_next_frame"] is not None:
                        predicted_next_frame = aux_infos["predicted_next_frame"][:-1, :, :, :] # (150-1, 128, 128, 4)
                        predicted_next_frame = predicted_next_frame[actions_batch.view(-1,)[:-1] != 0]
                        gt_next_frame = torch.cat([obs_batch["rgb"] / 255, obs_batch["depth"]], dim=3)[1:, :, :, :] # (150-1, 128, 128, 4)
                        gt_next_frame = gt_next_frame[actions_batch.view(-1,)[:-1] != 0]
                        iprl_loss += self.predict_next_frame_loss(predicted_next_frame, gt_next_frame)
                    
                    if aux_infos["predicted_next_spectrogram"] is not None:
                        predicted_next_spectrogram = aux_infos["predicted_next_spectrogram"][:-1, :, :, :]
                        predicted_next_spectrogram = predicted_next_spectrogram[actions_batch.view(-1,)[:-1] != 0]
                        gt_next_spectrogram = obs_batch["spectrogram"][1:, :, :, :]
                        gt_next_spectrogram = gt_next_spectrogram[actions_batch.view(-1,)[:-1] != 0]
                        iprl_loss += self.predict_next_spectrogram_loss(predicted_next_spectrogram, gt_next_spectrogram)
                    
                    if aux_infos["predicted_semantic"] is not None:
                        predicted_semantic = aux_infos["predicted_semantic"]
                        gt_semantic = obs_batch["semantic"]
                        iprl_loss += self.predict_semantic_loss(predicted_semantic, gt_semantic)
                    
                    if aux_infos["predicted_audio_location"] is not None:
                        predicted_audio_location = aux_infos["predicted_audio_location"]
                        gt_audio_location = obs_batch["pointgoal_with_gps_compass"].unsqueeze(0)
                        iprl_loss += self.predict_audio_location_loss(predicted_audio_location, gt_audio_location)
                    
                    if aux_infos["predicted_audio_category"] is not None:
                        predicted_audio_category = aux_infos["predicted_audio_category"]
                        predicted_audio_category = predicted_audio_category.view(-1, predicted_audio_category.shape[-1])
                        gt_audio_category = torch.argmax(obs_batch["category"], dim=1).reshape(-1).long()
                        iprl_loss += self.predict_audio_category_loss(predicted_audio_category, gt_audio_category)
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

        self.update_cnt += 1

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
