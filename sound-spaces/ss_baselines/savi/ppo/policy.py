#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import abc
import logging
import itertools
import sys
import gc

import torch
import numpy as np
import torch.nn as nn
from torchsummary import summary

from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, LocationBelief, CategoryBelief, Category
from ss_baselines.common.utils import CategoricalNet
from ss_baselines.av_nav.models.rnn_state_encoder import RNNStateEncoder
from ss_baselines.savi.models.visual_cnn import VisualCNN
from ss_baselines.savi.models.audio_cnn import AudioCNN
from ss_baselines.savi.models.smt_state_encoder import SMTStateEncoder
from ss_baselines.savi.models.smt_cnn import SMTCNN
from ss_baselines.savi.models.direct_map_encoder import DirectMapEncoder
from ss_baselines.savi.models.instruction_predictor import InstructionPredictor

from ss_baselines.savi.models.auxiliary_module import (
    ProgressMonitorPredictor,
    NextOracleActionPredictor,
    NextFramePredictor,
    NextSpectrogramPredictor,
    SemanticPredictor,
    AudioLocationPredictor,
    AudioCategoryPredictor,
)

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.lang import R2RLang
from xgenerator.common.model import VisualImageEncoder

DUAL_GOAL_DELIMITER = ','


class Policy(nn.Module):
    def __init__(self, net, dim_actions):
        super().__init__()
        self.net = net
        self.dim_actions = dim_actions
        self.use_iprl = False

        self.action_distribution = CategoricalNet(
            self.net.output_size, self.dim_actions
        )
        self.critic = CriticHead(self.net.output_size)

    def forward(self, *x):
        raise NotImplementedError

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_direct_map,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        need_logits=False,
        deterministic=False,
    ):
        iprl_logits = None
        if self.use_iprl:
            features, rnn_hidden_states, ext_memory_feats, direct_map, iprl_logits = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks, need_logits
            )
        else:
            features, rnn_hidden_states, ext_memory_feats, direct_map = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks
            )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        if deterministic:
            action = distribution.mode()
        else:
            action = distribution.sample()

        action_log_probs = distribution.log_probs(action)

        return value, action, action_log_probs, rnn_hidden_states, ext_memory_feats, direct_map, iprl_logits

    def get_value(self, observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks):
        if self.use_iprl:
            features, _, _ , _, _ = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks, False
            )
        else:
            features, _, _ , _ = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks
            )
        return self.critic(features)

    def evaluate_actions(
        self,
        observations,
        rnn_hidden_states,
        prev_direct_map,
        prev_actions,
        masks,
        action,
        ext_memory,
        ext_memory_masks,
    ):
        iprl_logits = None
        if self.use_iprl:
            features, rnn_hidden_states, ext_memory_feats, direct_map, iprl_logits = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions,
                masks, ext_memory, ext_memory_masks, True, action,
            )
        else:
            features, rnn_hidden_states, ext_memory_feats, direct_map = self.net(
                observations, rnn_hidden_states, prev_direct_map, prev_actions,
                masks, ext_memory, ext_memory_masks
            )
        distribution = self.action_distribution(features)
        value = self.critic(features)

        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy().mean()

        return value, action_log_probs, distribution_entropy, rnn_hidden_states, ext_memory_feats, direct_map, iprl_logits


class CriticHead(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        return self.fc(x)


class AudioNavBaselinePolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        goal_sensor_uuid,
        hidden_size=512,
        extra_rgb=False,
        use_mlp_state_encoder=False
    ):
        super().__init__(
            AudioNavBaselineNet(
                observation_space=observation_space,
                hidden_size=hidden_size,
                goal_sensor_uuid=goal_sensor_uuid,
                extra_rgb=extra_rgb,
                use_mlp_state_encoder=use_mlp_state_encoder
            ),
            action_space.n,
        )


class AudioNavSMTPolicy(Policy):
    def __init__(self, observation_space, action_space, direct_map_size, goal_num, hidden_size=128, **kwargs):
        super().__init__(
            AudioNavSMTNet(
                observation_space,
                action_space,
                direct_map_size,
                goal_num,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )

class IPRLAudioNavSMTPolicy(Policy):
    def __init__(
        self,
        observation_space,
        action_space,
        direct_map_size,
        goal_num,
        hidden_size=128,
        **kwargs
    ):
        super().__init__(
            IPRLAudioNavSMTNet(
                observation_space,
                action_space,
                direct_map_size,
                goal_num,
                hidden_size=hidden_size,
                **kwargs
            ),
            action_space.n
        )
        self.use_iprl = True


class Net(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, observations, rnn_hidden_states, prev_actions, masks):
        pass

    @property
    @abc.abstractmethod
    def output_size(self):
        pass

    @property
    @abc.abstractmethod
    def num_recurrent_layers(self):
        pass

    @property
    @abc.abstractmethod
    def is_blind(self):
        pass


class AudioNavBaselineNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    def __init__(self, observation_space, hidden_size, goal_sensor_uuid, extra_rgb=False, use_mlp_state_encoder=False):
        super().__init__()
        self.goal_sensor_uuid = goal_sensor_uuid
        self._hidden_size = hidden_size
        self._audiogoal = False
        self._pointgoal = False
        self._n_pointgoal = 0
        self._label = 'category' in observation_space.spaces

        # for goal descriptors
        self._use_label_belief = False
        self._use_location_belief = False
        self._use_mlp_state_encoder = use_mlp_state_encoder

        if DUAL_GOAL_DELIMITER in self.goal_sensor_uuid:
            goal1_uuid, goal2_uuid = self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)
            self._audiogoal = self._pointgoal = True
            self._n_pointgoal = observation_space.spaces[goal1_uuid].shape[0]
        else:
            if 'pointgoal_with_gps_compass' == self.goal_sensor_uuid:
                self._pointgoal = True
                self._n_pointgoal = observation_space.spaces[self.goal_sensor_uuid].shape[0]
            else:
                self._audiogoal = True

        self.visual_encoder = VisualCNN(observation_space, hidden_size, extra_rgb)
        if self._audiogoal:
            if 'audiogoal' in self.goal_sensor_uuid:
                audiogoal_sensor = 'audiogoal'
            elif 'spectrogram' in self.goal_sensor_uuid:
                audiogoal_sensor = 'spectrogram'
            self.audio_encoder = AudioCNN(observation_space, hidden_size, audiogoal_sensor)

        rnn_input_size = (0 if self.is_blind else self._hidden_size) + \
                         (self._n_pointgoal if self._pointgoal else 0) + \
                         (self._hidden_size if self._audiogoal else 0) + \
                         (observation_space.spaces['category'].shape[0] if self._label else 0) + \
                         (observation_space.spaces[CategoryBelief.cls_uuid].shape[0] if self._use_label_belief else 0) + \
                         (observation_space.spaces[LocationBelief.cls_uuid].shape[0] if self._use_location_belief else 0)
        if not self._use_mlp_state_encoder:
            self.state_encoder = RNNStateEncoder(rnn_input_size, self._hidden_size)
        else:
            self.state_encoder = nn.Linear(rnn_input_size, self._hidden_size)

        if not self.visual_encoder.is_blind:
            summary(self.visual_encoder.cnn, self.visual_encoder.input_shape, device='cpu')
        if self._audiogoal:
            audio_shape = observation_space.spaces[audiogoal_sensor].shape
            summary(self.audio_encoder.cnn, (audio_shape[2], audio_shape[0], audio_shape[1]), device='cpu')

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        if self._use_mlp_state_encoder:
            return 1
        else:
            return self.state_encoder.num_recurrent_layers

    def forward(self, observations, rnn_hidden_states, prev_actions, masks, ext_memory=None, ext_memory_masks=None):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
        if self._label:
            x.append(observations['category'].to(device=x[0].device))

        if self._use_label_belief:
            x.append(observations[CategoryBelief.cls_uuid])
        if self._use_location_belief:
            x.append(observations[LocationBelief.cls_uuid])

        x1 = torch.cat(x, dim=1)
        if self._use_mlp_state_encoder:
            x2 = self.state_encoder(x1)
            rnn_hidden_states1 = x2
        else:
            x2, rnn_hidden_states1 = self.state_encoder(x1, rnn_hidden_states, masks)

        assert not torch.isnan(x2).any().item()

        return x2, rnn_hidden_states1, None

    def get_features(self, observations, prev_actions):
        x = []

        if self._pointgoal:
            x.append(observations[self.goal_sensor_uuid.split(DUAL_GOAL_DELIMITER)[0]])
        if self._audiogoal:
            x.append(self.audio_encoder(observations))
        if not self.is_blind:
            x.append(self.visual_encoder(observations))
        if self._label:
            x.append(observations['category'].to(device=x[0].device))

        if self._use_label_belief:
            x.append(observations[CategoryBelief.cls_uuid])
        if self._use_location_belief:
            x.append(observations[LocationBelief.cls_uuid])

        x = torch.cat(x, dim=1)

        return x


class AudioNavSMTNet(Net):
    r"""Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN. Implements the
    policy from Scene Memory Transformer: https://arxiv.org/abs/1903.03878
    """

    def __init__(
        self,
        observation_space,
        action_space,
        direct_map_size,
        goal_num,
        hidden_size=128,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        use_xgen_visual_encoder=False,
        **kwargs
    ):
        super().__init__()
        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input
        self.direct_map_size = direct_map_size
        self.goal_num = goal_num
        self.use_xgen_visual_encoder = use_xgen_visual_encoder

        self.use_audio = SpectrogramSensor.cls_uuid in observation_space.spaces
        # assert SpectrogramSensor.cls_uuid in observation_space.spaces
        if self.use_audio:
            self.goal_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid)
            audio_feature_dims = 128
        else:
            audio_feature_dims = 0

        if self.use_xgen_visual_encoder:
            h, w, _ = observation_space["rgb"].shape
            self.visual_encoder = VisualImageEncoder((h, w, 4), 512-4)
        else:
            self.visual_encoder = SMTCNN(observation_space)
        
        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0
        
        if direct_map_size is not None:
            self.direct_map_encoder = DirectMapEncoder(
                input_size=audio_feature_dims + self._action_size + direct_map_size,
                output_size=direct_map_size,
                action_num=self._action_size,
            )
        
        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims + (
            direct_map_size if direct_map_size is not None else 0
        )

        if self._use_category_input:
            nfeats += 21

        # Add pose observations to the memory
        assert PoseSensor.cls_uuid in observation_space.spaces
        if PoseSensor.cls_uuid in observation_space.spaces:
            pose_dims = observation_space.spaces[PoseSensor.cls_uuid].shape[0]
            # Specify which part of the memory corresponds to pose_dims
            pose_indices = (nfeats, nfeats + pose_dims)
            nfeats += pose_dims
        else:
            pose_indices = None

        self._feature_size = nfeats

        self.smt_state_encoder = SMTStateEncoder(
            input_size=nfeats,
            dim_feedforward=hidden_size,
            pose_indices=pose_indices,
            **kwargs,
        )

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

        self.train()

    @property
    def memory_dim(self):
        return self._feature_size

    @property
    def output_size(self):
        size = self.smt_state_encoder.hidden_state_size
        if self._use_residual_connection:
            size += self._feature_size
        return size

    @property
    def is_blind(self):
        return False

    @property
    def num_recurrent_layers(self):
        return -1

    def forward(self, observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks, need_enc_memory=False):
        x, direct_map = self.get_features(observations, prev_direct_map, prev_actions)

        if self._use_belief_as_goal:
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            if self._use_label_belief:
                if self._normalize_category_distribution:
                    belief[:, :21] = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1)
                else:
                    belief[:, :21] = observations[CategoryBelief.cls_uuid]

            if self._use_location_belief:
                belief[:, 21:21 + 2*self.goal_num] = observations[LocationBelief.cls_uuid]

            if self._use_belief_encoder:
                belief = self.belief_encoder(belief)
        else:
            belief = None

        if ObjectGoalSensor.cls_uuid in observations.keys():
            belief = torch.zeros((x.shape[0], self._hidden_size), device=x.device)
            objectgoal = observations[ObjectGoalSensor.cls_uuid]
            objectgoal = torch.nn.functional.one_hot(objectgoal.to(torch.int64).view(-1), num_classes=21)
            belief[:,:21] = objectgoal

        x_att, enc_memory = self.smt_state_encoder(x, ext_memory, ext_memory_masks, need_enc_memory=True, goal=belief)
        if self._use_residual_connection:
            x_att = torch.cat([x_att, x], 1)

        if need_enc_memory:
            return x_att, rnn_hidden_states, x, direct_map, belief, enc_memory
        else:
            return x_att, rnn_hidden_states, x, direct_map

    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def pretrained_initialization(self, path):
        logging.info(f'AudioNavSMTNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(path)['state_dict']
        cleaned_state_dict = {
            k[len('actor_critic.net.'):]: v for k, v in state_dict.items()
            if 'actor_critic.net.' in k
        }
        self.load_state_dict(cleaned_state_dict, strict=False)

    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMTNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        if self.use_audio:
            params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        if self.direct_map_size is not None:
            params_to_freeze.append(self.direct_map_encoder.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        if self.use_audio:
            self.goal_encoder.eval()
        self.visual_encoder.eval()
        if self.direct_map_size is not None:
            self.direct_map_encoder.eval()

    def get_features(self, observations, prev_direct_map, prev_actions):
        x = []
        if self.use_xgen_visual_encoder:
            rgb = observations["rgb"] / 255.0
            depth = observations["depth"]
            if len(np.shape(depth)) == 5:
                depth = np.squeeze(depth, axis=4)
            imgs = torch.cat([rgb, depth], axis=3)
            visual_features = self.visual_encoder(imgs)
        else:
            rgb = observations["rgb"]
            depth = observations["depth"]
            visual_features = self.visual_encoder(observations)
        x.append(visual_features)
        x.append(self.action_encoder(self._get_one_hot(prev_actions)))
        if self.use_audio:
            x.append(self.goal_encoder(observations))
        if self.direct_map_size is not None:
            direct_map = self.direct_map_encoder(prev_direct_map, x[2], None, self._get_one_hot(prev_actions))
            x.append(direct_map)
        else:
            direct_map = None
        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=1)

        return x, direct_map


class IPRLAudioNavSMTNet(AudioNavSMTNet):
    def __init__(
        self,
        observation_space,
        action_space,
        direct_map_size,
        goal_num,
        iprl_max_instr_len,
        iprl_num_decoder_layers,
        iprl_vocab_emb_size,
        iprl_emb_size,
        iprl_nhead,
        iprl_dim_feedforward,
        iprl_dropout,
        iprl_use_gt_D,
        iprl_feedback,
        iprl_use_bos,
        use_instruction_predictor,
        use_progress_predictor,
        use_action_predictor,
        use_next_frame_predictor,
        use_next_spectrogram_predictor,
        use_semantic_predictor,
        use_audio_location_predictor,
        use_audio_category_predictor,
        hidden_size=128,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        use_xgen_visual_encoder=False,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            direct_map_size,
            goal_num,
            hidden_size,
            False, # use_pretrained (superの中では呼ばない)
            pretrained_path,
            use_belief_as_goal,
            use_label_belief,
            use_location_belief,
            use_belief_encoding,
            normalize_category_distribution,
            use_category_input,
            use_xgen_visual_encoder,
            **kwargs,
        )
        lang = R2RLang(name="r2r_train")

        self.use_instruction_predictor = use_instruction_predictor
        self.use_progress_predictor = use_progress_predictor
        self.use_action_predictor = use_action_predictor
        self.use_next_frame_predictor = use_next_frame_predictor
        self.use_next_spectrogram_predictor = use_next_spectrogram_predictor
        self.use_semantic_predictor = use_semantic_predictor
        self.use_audio_location_predictor = use_audio_location_predictor
        self.use_audio_category_predictor = use_audio_category_predictor

        if self.use_instruction_predictor:
            self.instruction_predictor = InstructionPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                vocab_emb_size=iprl_vocab_emb_size,
                emb_size=iprl_emb_size,
                max_instr_len=iprl_max_instr_len,
                nhead=iprl_nhead,
                vocab_size=lang.vocab_size,
                glove=lang.glove_vec,
                dim_feedforward=iprl_dim_feedforward,
                dropout=iprl_dropout,
                pretraining=kwargs["pretraining"],
                use_bos=iprl_use_bos,
            )
        if self.use_progress_predictor:
            self.progress_predictor = ProgressMonitorPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        if self.use_action_predictor:
            self.action_predictor = NextOracleActionPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        if self.use_next_frame_predictor:
            self.next_frame_predictor = NextFramePredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        if self.use_next_spectrogram_predictor:
            self.next_spectrogram_predictor = NextSpectrogramPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        if self.use_semantic_predictor:
            raise NotImplementedError()
        if self.use_audio_location_predictor:
            self.audio_location_predictor = AudioLocationPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        if self.use_audio_category_predictor:
            self.audio_category_predictor = AudioCategoryPredictor(
                num_decoder_layers=iprl_num_decoder_layers,
                emb_size=iprl_emb_size,
                nhead=iprl_nhead,
                dim_feedforward=iprl_dim_feedforward,
                pretraining=kwargs["pretraining"],
            )
        
        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)
        self.iprl_use_gt_D = iprl_use_gt_D
        self.feedback = iprl_feedback



        self.train()
    
    def pretrained_initialization(self, path):
        logging.info(f'AudioNavSMTNet ===> Loading pretrained model from {path}')
        state_dict = torch.load(
            path,
            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
        )
        if "xgenerator" in path:
            cleaned_state_dict = {}
            for k, v in state_dict.items():
                if "encoder" in k:
                    cleaned_state_dict[f"smt_state_encoder.{k}"] = v
                elif "decoder" in k:
                    if not ".norm." in k:
                        cleaned_state_dict[f"instruction_predictor.{k[len('transformer.'):]}"] = v
                elif "generator" in k or "word_vocab_emb" in k or "word_emb" in k:
                    cleaned_state_dict[f"instruction_predictor.{k}"] = v
                elif self.use_xgen_visual_encoder and ("visual_emb" in k):
                    if k == "visual_emb.cnn.0.weight": # torch.Size([32, 7, 8, 8])
                        cleaned_state_dict[f"visual_encoder.{k[len('visual_emb.'):]}"] = v[:, :4, :, :]
                    else:
                        cleaned_state_dict[f"visual_encoder.{k[len('visual_emb.'):]}"] = v
                else:
                    continue
        else:
            cleaned_state_dict = {}
            for k, v in state_dict["state_dict"].items():
                if "actor_critic.net." in k:
                    k = k[len("actor_critic.net."):]
                cleaned_state_dict[k] = v
        
        self.load_state_dict(cleaned_state_dict, strict=False)
    
    def forward(
        self,
        observations,
        rnn_hidden_states,
        prev_direct_map,
        prev_actions,
        masks,
        ext_memory,
        ext_memory_masks,
        need_logits=False,
        actions=None,
    ):
        x_att, rnn_hidden_states, x, direct_map, belief, enc_memory = super().forward(
            observations, rnn_hidden_states, prev_direct_map, prev_actions, masks, ext_memory, ext_memory_masks, True,
        )
        if not need_logits:
            return x_att, rnn_hidden_states, x, direct_map, None
        
        if self.iprl_use_gt_D:
            category = observations["category"] # (batch, 21)
            location = observations["pointgoal_with_gps_compass"] # (batch, 2)
        else:
            category = nn.functional.softmax(belief[:, :21], dim=1)
            location = belief[:, 21:21+2*self.goal_num]

        if self.use_instruction_predictor:
            if self.feedback == "teacher":
                if "generated_instruction" in observations.keys():
                    target = observations["generated_instruction"] # (batch, instr_len)
                elif "habitat_sim_generated_instruction" in observations.keys():
                    target = observations["habitat_sim_generated_instruction"] # (batch, instr_len)
                else:
                    raise Exception(f"feedback type is teacher, but there is no generated instruction.")
            elif self.feedback == "student":
                target = None
            else:
                raise Exception(f"feedback must be 'teacher' or 'student', not {self.feedback}")
            
            logits = self.instruction_predictor(
                category=category, # (batch, 21)
                location=location, # (batch, 2)
                target=target,
                memory=enc_memory, # (mem_size, batch, smt_hidden)
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            logits = None

        if self.use_action_predictor:
            predicted_action = self.action_predictor(
                memory=enc_memory,
                target=belief.unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_action = None
        
        if self.use_progress_predictor:
            predicted_progress = self.progress_predictor(
                memory=enc_memory,
                target=belief.unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_progress = None
        
        if self.use_next_frame_predictor:
            predicted_next_frame = self.next_frame_predictor(
                memory=enc_memory,
                action=self._get_one_hot(actions).unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_next_frame = None
        
        if self.use_next_spectrogram_predictor:
            predicted_next_spectrogram = self.next_spectrogram_predictor(
                memory=enc_memory,
                action=self._get_one_hot(actions).unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_next_spectrogram = None
        

        if self.use_semantic_predictor:
            raise NotImplementedError()
        else:
            predicted_semantic = None

        if self.use_audio_location_predictor:
            predicted_audio_location = self.audio_location_predictor(
                memory=enc_memory,
                target=belief.unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_audio_location = None
        
        if self.use_audio_category_predictor:
            predicted_audio_category = self.audio_category_predictor(
                memory=enc_memory,
                target=belief.unsqueeze(0),
                memory_key_padding_mask=(1 - ext_memory_masks) > 0,
            )
        else:
            predicted_audio_category = None
        
        aux_infos = {
            "logits": logits,
            "predicted_action": predicted_action,
            "predicted_progress": predicted_progress,
            "predicted_next_frame": predicted_next_frame,
            "predicted_next_spectrogram": predicted_next_spectrogram,
            "predicted_semantic": predicted_semantic,
            "predicted_audio_location": predicted_audio_location,
            "predicted_audio_category": predicted_audio_category,
        }

        return x_att, rnn_hidden_states, x, direct_map, aux_infos
