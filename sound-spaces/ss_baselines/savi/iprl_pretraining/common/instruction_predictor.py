import logging
import itertools
import sys

import torch
import torch.nn as nn
import numpy as np
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from soundspaces.tasks.nav import PoseSensor, SpectrogramSensor, LocationBelief, CategoryBelief, Category
from ss_baselines.savi.models.audio_cnn import AudioCNN
from ss_baselines.savi.models.smt_state_encoder import SMTStateEncoder
from ss_baselines.savi.models.smt_cnn import SMTCNN
from ss_baselines.savi.models.instruction_predictor import InstructionPredictor
from ss_baselines.savi.models.belief_predictor import BeliefPredictor

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.lang import R2RLang, VideoLLaMA2Lang, Qwen25VLLang
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
        max_grad_norm,
        use_xgen_visual_encoder,
        on_or_off="on",
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
        visual_encoder_output_size=512-4,
        tokenizer_type="r2r",
        share_decoder=False,
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
        self.goal_num = goal_num
        self.on_or_off = on_or_off
        self.use_xgen_visual_encoder = use_xgen_visual_encoder

        super().__init__()
        self.use_audio = SpectrogramSensor.cls_uuid in observation_space.spaces
        if self.use_audio:
            self.goal_encoder = AudioCNN(observation_space, 128, SpectrogramSensor.cls_uuid)
            audio_feature_dims = 128
        else:
            self.goal_encoder = None
            audio_feature_dims = 0

        if self.use_xgen_visual_encoder:
            h, w, _ = observation_space["rgb"].shape
            self.visual_encoder = VisualImageEncoder((h, w, 4), visual_encoder_output_size)
        else:
            self.visual_encoder = SMTCNN(observation_space)

        self.action_encoder = nn.Linear(self._action_size, 16)
        action_encoding_dims = 16

        nfeats = self.visual_encoder.feature_dims + action_encoding_dims + audio_feature_dims

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
            on_or_off=on_or_off,
        )
        
        if not iprl_use_gt_D:
            self.belief_predictor = BeliefPredictor(
                belief_config=belief_cfg,
                device=self.device,
                input_size=self.smt_state_encoder._input_size,
                pose_indices=self.smt_state_encoder._pose_indices,
                goal_num=1,
                hidden_state_size=self.smt_state_encoder.hidden_state_size,
                num_env=batch_size,
                has_distractor_sound=False,
            )

        if self._use_belief_encoder:
            self.belief_encoder = nn.Linear(self._hidden_size, self._hidden_size)

        self.tokenizer_type = tokenizer_type
        if tokenizer_type == "r2r":
            lang = R2RLang()
        elif tokenizer_type == "video_llama2":
            lang = VideoLLaMA2Lang()
        elif tokenizer_type == "qwen25vl":
            lang = Qwen25VLLang()
        else:
            raise Exception(f"tokenizer_type: {tokenizer_type}")
        self.instruction_predictor = InstructionPredictor(
            num_decoder_layers=iprl_num_decoder_layers,
            vocab_emb_size=lang.word_embed_size,
            emb_size=iprl_emb_size,
            max_instr_len=iprl_max_instr_len,
            nhead=iprl_nhead,
            vocab_size=lang.vocab_size,
            glove=lang.glove_vec,
            dim_feedforward=iprl_dim_feedforward,
            dropout=iprl_dropout,
            pretraining=kwargs["pretraining"],
            use_bos=iprl_use_bos,
            share_decoder=share_decoder,
        )
        self.iprl_use_gt_D = iprl_use_gt_D
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
            map_location=torch.device('cpu'),
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
            # cleaned_state_dict = {}
            # for k, v in state_dict['state_dict'].items():
            #     if 'actor_critic.net.' in k:
            #         cleaned_state_dict[k[len('actor_critic.net.'):]] = v
            #     elif 'actor_critic' in k:
            #         cleaned_state_dict[k[len('actor_critic.'):]] = v

            # XPredictorだけ別に読み込む
            # xpredictor_state_dict = torch.load(
            #     'data/models/ss1-savi/mp3d/ss1savi-savi-iprl-pre-offpolicy-past-v3/data/ckpt.138.pth',
            #     map_location=torch.device('cpu'),
            # )
            # for k, v in xpredictor_state_dict['state_dict'].items():
            #     if "instruction_predictor" in k:
            #         cleaned_state_dict[k] = v

        self.load_state_dict(cleaned_state_dict, strict=False)

        if not self.iprl_use_gt_D:
            # state_dict = torch.load(
            #     'data/models/ss1-savi/mp3d/ss1savi-savi-iprl-ft-future-obsenc-offplicy-tf-duration/data/ckpt.191.pth',
            #     map_location=torch.device('cpu'),
            # )
            self.belief_predictor.load_state_dict(state_dict["belief_predictor"])
    
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
        logging.info(f'AudioNavSMTNet ===> Freezing goal, visual, fusion encoders!')
        params_to_freeze = []
        if self.use_audio:
            params_to_freeze.append(self.goal_encoder.parameters())
        params_to_freeze.append(self.visual_encoder.parameters())
        params_to_freeze.append(self.action_encoder.parameters())
        for p in itertools.chain(*params_to_freeze):
            p.requires_grad = False

    def set_eval_encoders(self):
        """Sets the goal, visual and fusion encoders to eval mode."""
        if self.use_audio:
            self.goal_encoder.eval()
        self.visual_encoder.eval()

    def get_features(self, observations, prev_actions, return_visual_features=False):
        x = []
        
        if self.use_xgen_visual_encoder:
            if self.on_or_off == "on":
                raise NotImplementedError()
            rgb = observations["rgb"]
            depth = observations["depth"]
            imgs = torch.cat([rgb, depth], axis=4)
            L, N, H, W, C = imgs.size()
            imgs = imgs.view(L*N, H, W, C)
            visual_features = self.visual_encoder(imgs)
            visual_features = visual_features.view(L, N, -1)
        else:
            observations["rgb"] = observations["rgb"] * 255
            visual_features = self.visual_encoder(observations, self.on_or_off)
        x.append(visual_features)

        if self.on_or_off == "off":
            L, N, _ = prev_actions.size()
            prev_actions = prev_actions.view(L*N, 1)
            action_features = self.action_encoder(self._get_one_hot(prev_actions))
            action_features = action_features.view(L, N, 16)
        else:
            action_features = self.action_encoder(self._get_one_hot(prev_actions))
            
        x.append(action_features)
        if self.use_audio:
            x.append(self.goal_encoder(observations, self.on_or_off))

        if self._use_category_input:
            x.append(observations[Category.cls_uuid])

        x.append(observations[PoseSensor.cls_uuid])

        if self.on_or_off == "on":
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(x, dim=2)
        # f = open("debug.txt", "a")
        # f.write(f"x: {x.size()}\n")
        # f.close()

        if return_visual_features:
            return x, visual_features
        else:
            return x
    
    def _get_one_hot(self, actions):
        if actions.shape[1] == self._action_size:
            return actions
        else:
            N = actions.shape[0]
            actions_oh = torch.zeros(N, self._action_size, device=actions.device)
            actions_oh.scatter_(1, actions.long(), 1)
            return actions_oh

    def forward(self, observations, prev_actions, masks, ext_memory, ext_memory_masks, return_visual_features=False):
        # f = open("debug.txt", "a")
        # f.write(f"--------- FORWARD --------\n")
        # for k, v in observations.items():
        #     f.write(f"{k}: {v.size()}\n")
        # f.write(f"prev_actions: {prev_actions.size()}\n")
        # f.write(f"ext_memory_masks: {ext_memory_masks.size()}\n")
        # f.close()
        if self.on_or_off == "on":
            return self.on_forward(observations, prev_actions, masks, ext_memory, ext_memory_masks)
        elif self.on_or_off == "off":
            return self.off_forward(observations, prev_actions, ext_memory_masks, observations["target"], return_visual_features)
        else:
            raise Exception(f"on_or_off: {self.on_or_off}")
    
    def generate(
        self,
        observations,
        prev_actions,
        save_dir_path,
        beam_num,
        top_k,
        top_p,
        temperature,
    ):
        x = self.get_features(observations, prev_actions)
        _, enc_memory = self.smt_state_encoder(x, None, None, path_lens=observations["seq_lengths"])

        if self.use_audio:
            if self.iprl_use_gt_D:
                raise Exception(f"self.iprl_use_gt_D must be False.")
            else: 
                with torch.no_grad():
                    observations = self.update_belief(observations)
                category = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1) # (batch, 21)
                location = observations[LocationBelief.cls_uuid] # (batch, 2)
        else:
            assert ObjectGoalSensor.cls_uuid in observations.keys()
            category = observations[ObjectGoalSensor.cls_uuid] # (batch, 21)
            location = torch.from_numpy(np.zeros((len(category), 2))).to(category.device).float() # (batch, 2)
        
        batch_size = category.shape[0]
        assert batch_size == 1, "batch_size must be 1."
        tokens = self.instruction_predictor.generate(
            category=category, # (batch, 21)
            location=location, # (batch, 2)
            memory=enc_memory, # (mem_size, batch, smt_hidden)
            save_dir_path=save_dir_path,
            beam_num=beam_num,
            tokenizer_type=self.tokenizer_type,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
        )
        return tokens

    def off_forward(self, observations, prev_actions, ext_memory_masks, targets, return_visual_features):
        if return_visual_features:
            x, visual_features = self.get_features(observations, prev_actions, return_visual_features)
        else:
            x = self.get_features(observations, prev_actions, return_visual_features)
            
        _, enc_memory = self.smt_state_encoder(x, None, ext_memory_masks, path_lens=observations["seq_lengths"])

        if self.use_audio:
            if self.iprl_use_gt_D:
                category =  observations["category"] # (batch, 21)
                location =  observations["location"] # (batch, 2)
            else: 
                with torch.no_grad():
                    observations = self.update_belief(observations)
                category = nn.functional.softmax(observations[CategoryBelief.cls_uuid], dim=1) # (batch, 21)
                location = observations[LocationBelief.cls_uuid] # (batch, 2)
        else:
            assert ObjectGoalSensor.cls_uuid in observations.keys()
            category = observations[ObjectGoalSensor.cls_uuid] # (batch, 21)
            location = torch.from_numpy(np.zeros((len(category), 2))).to(category.device).float() # (batch, 2)

        if self.feedback == "teacher":
            pass
        elif self.feedback == "student":
            targets = None
        else:
            raise Exception(f"feedback must be 'teacher' or 'student', not {self.feedback}")
        logits = self.instruction_predictor(
            category=category, # (batch, 21)
            location=location, # (batch, 2)
            target=targets, # (batch, instr_len)
            memory=enc_memory, # (mem_size, batch, smt_hidden)
            memory_key_padding_mask=ext_memory_masks,
            convert_mask=False,
        )
        if return_visual_features:
            return logits, visual_features
        else:
            return logits

    def on_forward(self, observations, prev_actions, masks, ext_memory, ext_memory_masks):
        x = self.get_features(observations, prev_actions)

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
        
        _, enc_memory = self.smt_state_encoder(x, ext_memory, ext_memory_masks, need_enc_memory=True, goal=belief)

        if self.iprl_use_gt_D:
            category = observations["category"] # (batch, 21)
            location = observations["pointgoal_with_gps_compass"] # (batch, 2)
        else:
            # max_indices = torch.argmax(belief[:, :21], dim=1)
            # category = torch.nn.functional.one_hot(max_indices, num_classes=21).float().cuda()
            category = nn.functional.softmax(belief[:, :21], dim=1)
            location = belief[:, 21:21+2*self.goal_num]
        
        if self.feedback == "teacher":
            target = observations["oracle_action_generated_instruction"] # (batch, instr_len)
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

        return logits
    
    def update_belief(self, observations):
        seq_len, N, _, _, _ = observations["rgb"].shape
        for i in range(seq_len):
            observation = {
                SpectrogramSensor.cls_uuid: observations[SpectrogramSensor.cls_uuid][i],
                'pose': observations['pose'][i],
                LocationBelief.cls_uuid: torch.zeros((N, 2)),
                CategoryBelief.cls_uuid: torch.zeros((N, 21)),
            }
            
            if i == 0:
                dones = [True for _ in range(N)]
            else:
                dones = [False for _ in range(N)]
            
            self.belief_predictor.update(observation, dones)
        
        observations[LocationBelief.cls_uuid] = observation[LocationBelief.cls_uuid].to("cuda")
        observations[CategoryBelief.cls_uuid] = observation[CategoryBelief.cls_uuid].to("cuda")
        return observations

    def before_step(self):
        nn.utils.clip_grad_norm_(
            self.parameters(), self.max_grad_norm
        )


class AudioNavSMTInstructionPredictorDDP(AudioNavSMTInstructionPredictor, DecentralizedDistributedMixinInstruction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
