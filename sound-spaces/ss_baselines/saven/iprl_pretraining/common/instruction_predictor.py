import logging
import itertools
import sys
import pickle

import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, LocationBelief, CategoryBelief, Category, KSAVENCategoryBelief
from ss_baselines.savi.models.audio_cnn import AudioCNN
from ss_baselines.savi.models.smt_state_encoder import SMTStateEncoder
from ss_baselines.saven.models.smt_cnn import SMTCNN, SMTCNN_saven, VisionPredictor
from ss_baselines.saven.models.gcn import GCN, DGL_GCN
from ss_baselines.savi.models.instruction_predictor import InstructionPredictor
from ss_baselines.saven.models.belief_predictor import BeliefPredictor

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.lang import R2RLang
from xgenerator.common.model import VisualImageEncoder


class DecentralizedDistributedMixinInstruction:
    def init_distributed(self, find_unused_params: bool = True) -> None:
        r"""Initializes distributed training for the model

        1. Broadcasts the model weights from world_rank 0 to all other workers
        2. Adds gradient hooks to the model

        :param find_unused_params: Whether or not to filter out unused parameters
                                   before gradient reduction.  This *must* be True if
                                   there are any parameters in the model that where unused in the
                                   forward pass, otherwise the gradient reduction
                                   will not work correctly.
        """
        # NB: Used to hide the hooks from the nn.Module,
        # so they don't show up in the state_dict
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model, device_ids=[device], output_device=device
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self, self.device)

        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = find_unused_params

    def before_backward(self, loss):
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class AudioNavSMTInstructionPredictor(nn.Module):
    def __init__(
        self,
        device,
        observation_space,
        action_space,
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
        max_grad_norm,
        hidden_size=128,
        use_pretrained=False,
        pretrained_path='',
        use_belief_as_goal=True,
        use_label_belief=True,
        use_location_belief=True,
        use_belief_encoding=False,
        normalize_category_distribution=False,
        use_category_input=False,
        belief_cfg=None,
        batch_size=-1,
        **kwargs,
    ):
        self.device = device

        self._use_action_encoding = True
        self._use_residual_connection = False
        self._use_belief_as_goal = use_belief_as_goal
        self._use_label_belief = use_label_belief
        self._use_location_belief = use_location_belief
        self.max_grad_norm = max_grad_norm
        self._hidden_size = hidden_size
        self._action_size = action_space.n
        self._use_belief_encoder = use_belief_encoding
        self._normalize_category_distribution = normalize_category_distribution
        self._use_category_input = use_category_input

        super().__init__()
        
        self.visual_encoder = SMTCNN_saven(observation_space)

        self.audio_gcn = GCN()  # DGL_GCN() GCN()
        self.visual_gcn = GCN()  # DGL_GCN() GCN()
        self.vision_predictor = VisionPredictor()

        if self._use_action_encoding:
            self.action_encoder = nn.Linear(self._action_size, 16)
            action_encoding_dims = 16
        else:
            action_encoding_dims = 0

        nfeats = self.visual_encoder.feature_dims + self.visual_gcn.feature_dims + action_encoding_dims

        if self._use_category_input:
            nfeats += 21
        
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
            nhead=kwargs["nhead"],
            num_encoder_layers=kwargs["num_encoder_layers"],
            num_decoder_layers=kwargs["num_decoder_layers"],
            dim_feedforward=hidden_size,
            dropout=kwargs["dropout"],
            activation=kwargs["activation"],
            pose_indices=pose_indices,
            pretraining=kwargs["pretraining"],
            on_or_off="off",
        )

        if not iprl_use_gt_D:
            raise NotImplementedError()
            # self.belief_predictor = BeliefPredictor(
            #     belief_cfg,
            #     self.device,
            #     self.smt_state_encoder._input_size,
            #     self.smt_state_encoder._pose_indices,
            #     self.smt_state_encoder.hidden_state_size,
            #     batch_size, # num_envs
            #     False, # has_distractor_sound
            # ).to(device=self.device)
            

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        mp3d_objects_of_interest_filepath = r"data/metadata/mp3d_objects_of_interest_data.bin"
        with open(mp3d_objects_of_interest_filepath, 'rb') as bin_file:
            self.ooi_objects_id_name = pickle.load(bin_file)
            self.ooi_regions_id_name = pickle.load(bin_file)
        
        graph_filename = r"data/metadata/mp3d_graph_object.csv"
        df = pd.read_csv(graph_filename, delimiter=',')
        self.object_regions = {}
        for index, row in df.iterrows():
            obj = row['Sounding Objects']
            regions = row['Regions']
            regions = [reg.strip() for reg in regions.split(',')]
            self.object_regions[obj] = regions

        lang = R2RLang(name="r2r_train")
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
            belief_dim=self.audio_gcn.feature_dims+2,
            share_decoder=False,
        )

        self.feedback = iprl_feedback

        self.optimizer = None

        if use_pretrained:
            assert(pretrained_path != '')
            self.pretrained_initialization(pretrained_path)

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
            cleaned_state_dict = state_dict['state_dict']
        self.load_state_dict(cleaned_state_dict, strict=False)
    
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
    
    def freeze_encoders(self):
        """Freeze goal, visual and fusion encoders. Pose encoder is not frozen."""
        logging.info(f'AudioNavSMTNet ===> Freezing visual_encoder and visual_gcn encoders!')
        params_to_freeze = []
        # params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        params_to_freeze.append(self.visual_gcn.parameters())
        if self._use_action_encoding:
            params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        # self.goal_encoder.eval()
        self.visual_encoder.eval()
        self.visual_gcn.eval()

    def get_features(self, observations, prev_actions):

        x = []
        
        observations["rgb"] = observations["rgb"] * 255
        x.append(self.visual_encoder(observations, "off"))

        L, N, _ = prev_actions.size()
        prev_actions = prev_actions.view(L*N, 1)
        action_features = self.action_encoder(self._get_one_hot(prev_actions))
        action_features = action_features.view(L, N, 16)
        x.append(action_features)

        obs_cat_predicts = self.vision_predictor(observations, "off") # (L, N, vis_dim)
        L, N, _ = obs_cat_predicts.shape
        visual_gcn_embds = torch.zeros(
            (obs_cat_predicts.shape[0], obs_cat_predicts.shape[1], self.visual_gcn.feature_dims),
            device=prev_actions.device,
        ) # (L, N, fea_dim)
        for i in range(len(obs_cat_predicts)):
            for j in range(len(obs_cat_predicts[i])):
                visual_gcn_embds[i, j, :] = self.visual_gcn(obs_cat_predicts[i, j])
        x.append(visual_gcn_embds)

        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        x = torch.cat(x, dim=2)

        return x
    
    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def forward(self, observations, prev_actions, masks, ext_memory, ext_memory_masks):
        return self.off_forward(observations, prev_actions, ext_memory_masks, observations["target"])

    def off_forward(self, observations, prev_actions, ext_memory_masks, targets):
        with torch.no_grad():
            x = self.get_features(observations, prev_actions)

        _, enc_memory = self.smt_state_encoder(x, None, ext_memory_masks, path_lens=observations["seq_lengths"])

        belief = torch.zeros((x.shape[1], self._hidden_size), device=x.device)
        
        # with torch.no_grad():
        #     observations = self.update_belief(observations)

        categories = observations["category"]
        obs_cat_belief = torch.zeros(
            (categories.shape[0], len(self.ooi_objects_id_name) + len(self.ooi_regions_id_name)),
            device=x.device,
        )
        object_ids = torch.argmax(categories, axis=1)
        for i, object_id in enumerate(object_ids):
            object_name = self.ooi_objects_id_name[object_id.item()]
            regions = self.object_regions[object_name]
            regions_id = [list(self.ooi_regions_id_name.keys())[list(self.ooi_regions_id_name.values()).index(reg)]
                              for reg in regions]
            regions_id = torch.tensor([1 if reg_id in regions_id else 0
                                       for reg_id in range(len(self.ooi_regions_id_name))])
            obs_cat_belief[i] = torch.cat([categories[i].to(x.device), regions_id.to(x.device)])
    
        audio_gcn_embds = torch.zeros((obs_cat_belief.shape[0], self.audio_gcn.feature_dims), device=x.device) # dim: 254
        for i in range(len(obs_cat_belief)):
            audio_gcn_embds[i, :] = self.audio_gcn(obs_cat_belief[i].to(x.device))
        belief[:, :self.audio_gcn.feature_dims] = audio_gcn_embds

        belief[:, self.audio_gcn.feature_dims:self.audio_gcn.feature_dims+2] = observations["location"].to(x.device)

        if self._use_belief_encoder:
            belief = self.belief_encoder(belief)

        category = belief[:, :self.audio_gcn.feature_dims]
        location = belief[:, self.audio_gcn.feature_dims:self.audio_gcn.feature_dims+2]

        if self.feedback == "teacher":
            pass
        elif self.feedback == "student":
            targets = None
        else:
            raise Exception(f"feedback must be 'teacher' or 'student', not {self.feedback}")
        logits = self.instruction_predictor(
            category=category, # (batch, 254)
            location=location, # (batch, 2)
            target=targets, # (batch, instr_len)
            memory=enc_memory, # (mem_size, batch, smt_hidden)
            memory_key_padding_mask=ext_memory_masks,
            convert_mask=False,
        )
        return logits
    
    def update_belief(self, observations):
        seq_len, N, _, _, _ = observations["rgb"].shape
        for i in range(seq_len):
            observation = {
                SpectrogramSensor.cls_uuid: observations[SpectrogramSensor.cls_uuid][i],
                'pose': observations['pose'][i],
                LocationBelief.cls_uuid: torch.zeros((N, 2)),
                KSAVENCategoryBelief.cls_uuid: torch.zeros((N, 45)),
            }
            
            if i == 0:
                dones = [True for _ in range(N)]
            else:
                dones = [False for _ in range(N)]
            
            self.belief_predictor.update(observation, dones)
        observations[LocationBelief.cls_uuid] = observation[LocationBelief.cls_uuid]
        observations[KSAVENCategoryBelief.cls_uuid] = observation[KSAVENCategoryBelief.cls_uuid]
        return observations

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.parameters(), self.max_grad_norm
        )


class AudioNavSMTInstructionPredictorDDP(AudioNavSMTInstructionPredictor, DecentralizedDistributedMixinInstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
