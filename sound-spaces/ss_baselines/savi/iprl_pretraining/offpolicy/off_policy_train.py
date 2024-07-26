import os
import sys
import time
import logging
import argparse
import contextlib


import yaml
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import all_reduce
from transformers.utils import logging as tf_logging
from gym import spaces

tf_logging.set_verbosity_error() 

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

from ss_baselines.savi.iprl_pretraining.common.instruction_predictor import AudioNavSMTInstructionPredictor
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import IPRLPretrainingDataset, CollateFn, compute_spectrogram
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_lmdb_dataset import IPRLPretrainingLMDBDataset
from habitat_baselines.config.default import get_config as habitat_get_config
from ss_baselines.savi.config.default import get_config as ss_get_config
from ss_baselines.savi.iprl_pretraining.common.videollama2_kd_loss import VideoLLaMA2KDLoss, R2RTokenizerVideoLLaMA2KDLoss

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX
from xgenerator.common.lang import tokens2sentences, sentence2token, R2RLang, VIDEO_LLAMA2_TOKENIZER


class OffPolicyEPRLPreTrainer():
    def __init__(
        self,
        config,
        logger,
        gpu_id,
        model_dir,
        use_lmdb_dataset,
        log_interval,
        save_interval,
        val_interval,
        environment_type,
    ):
        self.config = config
        self.gpu_id = gpu_id
        self.device = torch.device("cuda", self.gpu_id)
        self.logger = logger
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.val_interval = val_interval
        
        tb_log_dir = f"{model_dir}/tb"
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_log_dir) if int(os.environ["LOCAL_RANK"]) == 0 else contextlib.suppress()

        self.force_student = False
        self.n_update = 0
        self.foundation_model_type = self.config.FOUNDATION_MODEL.model_type
        self.foundation_model_path = self.config.FOUNDATION_MODEL.model_path
        self.tokenizer_type = self.config.TOKENIZER_TYPE
        if self.foundation_model_type == None:
            self.visual_encoder_output_size = 512-4
        elif self.foundation_model_type == "video_llama2":
            self.visual_encoder_output_size = self.config.FOUNDATION_MODEL.visual_encoder_output_size
        else:
            raise Exception(f"config.FOUNDATION_MODEL.model_type: {config.foundation_model_type}")
        self.environment_type = environment_type
        if self.environment_type == "ss1-savi":
            self.train_lmdb_dataset_path = "./data/lmdb_dataset/iprl_pretrain/train"
            self.train_data_num = 502103
            self.val_lmdb_dataset_path = "./data/lmdb_dataset/iprl_pretrain/val"
            self.val_data_num = 500
        elif self.environment_type == "habitat-objnav":
            self.train_lmdb_dataset_path = "habitat-lab/data/lmdb_dataset/iprl_pretrain/train"
            self.train_data_num = 51000
            self.val_lmdb_dataset_path = "habitat-lab/data/lmdb_dataset/iprl_pretrain/my_val"
            self.val_data_num = 500 # maxは800
        else:
            raise Exception(f"environemnt_type: {self.environment_type}")
        
        # prepare model and loss_fn
        self.instruction_predictor, self.loss_fn = self.setup_instruction_predictor()
        if self.foundation_model_type == "video_llama2" and self.tokenizer_type == "video_llama2":
            self.loss_fn = VideoLLaMA2KDLoss(
                visual_feature_coef=self.config.FOUNDATION_MODEL.visual_feature_coef,
                logits_coef=self.config.FOUNDATION_MODEL.logits_coef,
            )
        elif self.foundation_model_type == "video_llama2" and self.tokenizer_type == "r2r":
            self.loss_fn = R2RTokenizerVideoLLaMA2KDLoss(
                visual_feature_coef=self.config.FOUNDATION_MODEL.visual_feature_coef,
                logits_coef=self.config.FOUNDATION_MODEL.logits_coef,
            )
        
        # prepare datasets
        self.use_lmdb_dataset = use_lmdb_dataset
        self.train_dataloader, self.val_dataloader = self.setup_datasets()
        self.n_batch = len(self.train_dataloader)

        self.logger.info(f"device: {torch.device('cuda', self.gpu_id)}")

    def setup_instruction_predictor(self):
        spectrogram_shape = compute_spectrogram(np.ones((2, self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE))).shape
        if self.environment_type == "ss1-savi":
            observation_spaces = spaces.Dict({
                "pose": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "spectrogram": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=spectrogram_shape,
                    dtype=np.float32,
                ),
                "rgb": spaces.Box(
                    low=0,
                    high=1,
                    shape=(128, 128, 3),
                    dtype=np.float32,
                ),
                "depth": spaces.Box(
                    low=0,
                    high=1,
                    shape=(128, 128, 1),
                    dtype=np.float32,
                ),   
            })
        elif self.environment_type == "habitat-objnav":
            observation_spaces = spaces.Dict({
                "pose": spaces.Box(
                    low=np.finfo(np.float32).min,
                    high=np.finfo(np.float32).max,
                    shape=(4,),
                    dtype=np.float32,
                ),
                "rgb": spaces.Box(
                    low=0,
                    high=1,
                    shape=(480, 640, 3),
                    dtype=np.float32,
                ),
                "depth": spaces.Box(
                    low=0,
                    high=1,
                    shape=(480, 640, 1),
                    dtype=np.float32,
                ),   
            })
        else:
            raise Exception(f"environment_type: {self.environment_type}")

        action_spaces = spaces.Discrete(4)

        iprl_cfg=self.config.RL.PPO.INSTRUCTION_PREDICTOR
        ppo_cfg=self.config.RL.PPO
        smt_cfg=self.config.RL.PPO.SCENE_MEMORY_TRANSFORMER
        belief_cfg=self.config.RL.PPO.BELIEF_PREDICTOR
        has_distractor_sound=self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND
        pretrained=self.config.RL.DDPPO.pretrained
        pretrained_weights=self.config.RL.DDPPO.pretrained_weights

        instruction_predictor = AudioNavSMTInstructionPredictor(
            device=self.device,
            observation_space=observation_spaces,
            action_space=action_spaces,
            direct_map_size=None,
            goal_num=1,
            iprl_max_instr_len=iprl_cfg.max_instr_len,
            iprl_num_decoder_layers=iprl_cfg.num_decoder_layers,
            iprl_vocab_emb_size=iprl_cfg.vocab_emb_size,
            iprl_emb_size=iprl_cfg.emb_size,
            iprl_nhead=iprl_cfg.nhead,
            iprl_dim_feedforward=iprl_cfg.dim_feedforward,
            iprl_dropout=iprl_cfg.dropout,
            iprl_use_gt_D=iprl_cfg.iprl_use_gt_D,
            iprl_feedback=iprl_cfg.feedback,
            iprl_use_bos=iprl_cfg.use_bos,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_xgen_visual_encoder=smt_cfg.use_xgen_visual_encoder,
            on_or_off="off",
            hidden_size=smt_cfg.hidden_size,
            nhead=smt_cfg.nhead,
            num_encoder_layers=smt_cfg.num_encoder_layers,
            num_decoder_layers=smt_cfg.num_decoder_layers,
            dropout=smt_cfg.dropout,
            activation=smt_cfg.activation,
            use_pretrained=smt_cfg.use_pretrained,
            pretrained_path=smt_cfg.pretrained_path,
            pretraining=smt_cfg.pretraining,
            use_belief_encoding=smt_cfg.use_belief_encoding,
            use_belief_as_goal=ppo_cfg.use_belief_predictor,
            use_label_belief=belief_cfg.use_label_belief,
            use_location_belief=belief_cfg.use_location_belief,
            normalize_category_distribution=belief_cfg.normalize_category_distribution,
            use_category_input=has_distractor_sound,
            belief_cfg=belief_cfg,
            batch_size=ppo_cfg.num_steps,
            visual_encoder_output_size=self.visual_encoder_output_size,
            tokenizer_type=self.tokenizer_type,
        )
        instruction_predictor.optimizer = torch.optim.Adam(
            instruction_predictor.parameters(),
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
        )
        iprl_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        if smt_cfg.freeze_encoders:
            instruction_predictor.freeze_encoders()

        if pretrained:
            # load weights for both actor critic and the encoder
            pretrained_state = torch.load(pretrained_weights, map_location="cpu")
            instruction_predictor.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder" not in k and
                        "actor_critic.net.smt_state_encoder" not in k
                },
                strict=False
            )
            instruction_predictor.visual_encoder.rgb_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.rgb_encoder." in k
                },
            )
            instruction_predictor.visual_encoder.depth_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.depth_encoder." in k
                },
            )

        instruction_predictor.to(self.device)

        return instruction_predictor, iprl_loss_fn

    def setup_datasets(self):
        if multiprocessing.get_start_method() == 'fork':
            multiprocessing.set_start_method('spawn', force=True)
        if int(os.environ["LOCAL_RANK"]) == 0:
            self.logger.info("LOADING DATA...")
    
        if self.use_lmdb_dataset:
            train_split = self.config.TASK_CONFIG.DATASET.SPLIT
            if train_split == "train_w_instruction" or train_split == "train_w_past_instruction":
                train_dataset = IPRLPretrainingLMDBDataset(
                    self.config.TASK_CONFIG, train_split, self.train_lmdb_dataset_path, self.train_data_num,
                        self.foundation_model_type, self.foundation_model_path,
                        environment_type=self.environment_type,
                        device=self.device,
                )
            else:
                raise Exception(f"train_split: {train_split}")
            if "past" in train_split:
                val_split = "val_w_past_instruction"
            else:
                val_split = "val_w_instruction"
            if self.environment_type == "habitat-objnav":
                val_split = "my_" + val_split
            val_dataset = IPRLPretrainingLMDBDataset(
                self.config.TASK_CONFIG, val_split, self.val_lmdb_dataset_path, self.val_data_num,
                self.foundation_model_type, self.foundation_model_path,
                environment_type=self.environment_type,
                device=self.device,
            )
        else:
            # train_dataset = IPRLPretrainingDataset(config.TASK_CONFIG, config.SENSORS)
            raise NotImplementedError("use_lmdb_dataset must be True.") # val_datasetに対応させてないので

        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=int(os.environ["NP"]),
            shuffle=True,
            rank=self.gpu_id,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=int(os.environ["NP"]),
            shuffle=False,
            rank=self.gpu_id,
        )
        my_collate_fn = CollateFn(
            use_foundation_model=self.config.FOUNDATION_MODEL.model_type == "video_llama2",
            max_instr_len=self.config.FOUNDATION_MODEL.max_instr_len,
            environment_type=self.environment_type,
        )
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.RL.PPO.num_steps,
            drop_last=True,
            collate_fn=my_collate_fn,
            sampler=train_sampler,
            # num_workers=2,
        )
        val_dataloader = DataLoader(
            val_dataset,
            # num_workers=2,
            batch_size=self.config.RL.PPO.num_steps,
            drop_last=False,
            collate_fn=my_collate_fn,
            sampler=val_sampler,
        )
        n_batch = len(train_dataloader)
        if int(os.environ["LOCAL_RANK"]) == 0:
            self.logger.info(f"The number of train data: {len(train_dataset)}, batch: {n_batch}")
            self.logger.info("FINISH LOADING DATA!")
            self.logger.info("START TRAINING...")
        return train_dataloader, val_dataloader

    def save_checkpoint(self, file_name, extra_state=None) -> None: 
        checkpoint = {
            "state_dict": self.instruction_predictor.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            raise NotImplementedError()

        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def rollout(self, inputs, targets, visualize):
        if self.force_student:
            inputs["target"] = None
        else:
            if self.foundation_model_type is None:
                inputs["target"] = targets # (batch, instr_len)
            elif self.foundation_model_type == "video_llama2":
                # teahcer forcingなので、instructionsが必要
                if self.tokenizer_type == "r2r":
                    # まず、logitsを元にvideollama2のtokenierでsentenceに変換する
                    output_ids = torch.argmax(targets["logits"], dim=2) # (instr_len, batch)
                    instr_len, batch_size = np.shape(output_ids)
                    sentences = VIDEO_LLAMA2_TOKENIZER.batch_decode(output_ids.permute(1, 0), skip_special_tokens=True)

                    # 次に、sentenceを、r2rのtokenizerによってr2r用のtokenに変換する
                    # TODO ここ2重forなので遅くなっているはず
                    bos_idx = BOS_IDX
                    eos_idx = EOS_IDX
                    pad_idx = PAD_IDX
                    instructions = np.full((batch_size, instr_len), pad_idx)
                    for i, sentence in enumerate(sentences):
                        token = [bos_idx] + sentence2token(sentence) + [eos_idx]
                        if len(token) > instr_len:
                            token = token[:instr_len]
                        instructions[i, :len(token)] = token
                    
                    instructions = torch.from_numpy(instructions).cuda() # (batch, instr_len)
                elif self.tokenizer_type == "video_llama2":
                    instructions = torch.argmax(targets["logits"], dim=2).permute(1,0) # (batch, instr_len)
                    bos_idx = 1
                    batch_size = np.shape(instructions)[0]
                    instructions = torch.cat(
                        [torch.full((batch_size, 1), bos_idx).cuda(), instructions],
                        dim=1,
                    )[:, :-1] # (batch, instr_len)
                else:
                    raise Exception(f"tokenizer_type: {self.tokenizer_type}")
                inputs["target"] = instructions
            else:
                raise Exception(f"foundation_model_type: {self.foundation_model_type}")

        logits, visual_features = self.instruction_predictor(
            observations=inputs,
            prev_actions=inputs["action"],
            masks=None,
            ext_memory=None,
            ext_memory_masks=inputs["mask"],
            return_visual_features=True,
        )

        target_tokens = inputs["target"].permute(1, 0)[1:] # (instr_len-1, batch)
        if visualize and int(os.environ["LOCAL_RANK"]) == 0:
            if self.tokenizer_type == "r2r":
                lang = R2RLang()
                _, iprl_tokens = logits.max(2)
                pred_sentence = tokens2sentences(iprl_tokens, lang)[0]
                true_sentence = tokens2sentences(target_tokens, lang)[0]
            elif self.tokenizer_type == "video_llama2":
                _, iprl_tokens = logits.max(2) # iprl_tokens: (instr_len, batch)
                pred_sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(iprl_tokens.permute(1, 0), skip_special_tokens=True)[0]
                true_sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(target_tokens.permute(1, 0), skip_special_tokens=True)[0]
            else:
                raise Exception(f"tokenizer_type: {self.tokenizer_type}")

            logger.info(f"Pred: {pred_sentence}")
            logger.info(f"True: {true_sentence}")

        if self.foundation_model_type is None:
            targets = targets[:, 1:].permute(1, 0) # (instr_len-1, batch) BOSを除去
            loss = self.loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            info = None
        elif self.foundation_model_type == "video_llama2" and self.tokenizer_type == "video_llama2":
            loss, info = self.loss_fn(
                teacher_logits=targets["logits"][:-1],
                student_logits=logits,
                teacher_logits_mask=targets["logits_mask"][:, :-1],
                teacher_visual_features=targets["visual_features"],
                student_visual_features=visual_features,
                path_mask=inputs["mask"],
            )
        elif self.foundation_model_type == "video_llama2" and self.tokenizer_type == "r2r":
            loss, info = self.loss_fn(
                teacher_label=inputs["target"][:, 1:].permute(1, 0), # (instr_len-1, batch)
                student_logits=logits,
                teacher_visual_features=targets["visual_features"],
                student_visual_features=visual_features,
                path_mask=inputs["mask"],
            )
        else:
            raise Exception(f"foundation_model_type: {self.foundation_model_type}, tokenizer_type: {self.tokenizer_type}")

        return loss, info


    def evaluate(self):
        if int(os.environ["LOCAL_RANK"]) == 0:
            logger.info(f"========== Evaluation ==========")
        self.instruction_predictor.eval()
        s = time.time()
        losses = []
        visual_feature_losses = []
        logit_losses = []
        for i, (inputs, targets) in enumerate(self.val_dataloader):
            loss, info = self.rollout(inputs, targets, visualize=True)
            losses.append(loss.item())
            if self.foundation_model_type == "video_llama2":
                visual_feature_losses.append(info["visual_feature_loss"].item())
                logit_losses.append(info["logit_loss"].item())
        loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
        all_reduce(loss)
        if self.foundation_model_type == "video_llama2":
            visual_feature_loss = torch.from_numpy(np.array([np.mean(visual_feature_losses)])).to(gpu_id)
            logit_loss = torch.from_numpy(np.array([np.mean(logit_losses)])).to(gpu_id)
            all_reduce(visual_feature_loss)
            all_reduce(logit_loss)
        if int(os.environ["LOCAL_RANK"]) == 0:
            loss = loss.item() / int(os.environ["NP"])
            self.writer.add_scalar("eval/loss", loss, self.n_update)
            if self.foundation_model_type == "video_llama2":
                visual_feature_loss = visual_feature_loss.item() / int(os.environ["NP"])
                logit_loss = logit_loss.item() / int(os.environ["NP"])
                self.writer.add_scalar("eval/visual_feature_loss", visual_feature_loss, self.n_update)
                self.writer.add_scalar("eval/logit_loss", logit_loss, self.n_update)
                logger.info(f"visual_feature_loss: {visual_feature_loss:.5f}")
                logger.info(f"logit_loss: {logit_loss:.5f}")
            logger.info(f"Loss: {loss:.5f}")
            logger.info(f"Time: {(time.time() - s)/60:.2f} [min]")

    def train(self):
        self.instruction_predictor.train()

        n_loop = 0
        save_cnt = 0
        s = time.time()
        losses = []
        visual_feature_losses = []
        logit_losses = []
        while True:
            n_loop += 1
            for j, (inputs, targets) in enumerate(self.train_dataloader):
                self.n_update = self.n_batch*(n_loop-1) + j

                self.instruction_predictor.optimizer.zero_grad()
                loss, info = self.rollout(inputs, targets, visualize=False)
                loss.backward()
                self.instruction_predictor.optimizer.step()

                losses.append(loss.item())
                if info is not None:
                    visual_feature_losses.append(info["visual_feature_loss"].item())
                    logit_losses.append(info["logit_loss"].item())

                if self.n_update % self.log_interval == 0:
                    train_loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
                    all_reduce(train_loss)
                    train_loss = train_loss.item() / int(os.environ["NP"])
                    losses = []
                    if self.foundation_model_type == "video_llama2":
                        train_visual_feature_loss = torch.from_numpy(np.array([np.mean(visual_feature_losses)])).to(gpu_id)
                        train_logit_loss = torch.from_numpy(np.array([np.mean(logit_losses)])).to(gpu_id)
                        all_reduce(train_visual_feature_loss)
                        all_reduce(train_logit_loss)
                        train_visual_feature_loss = train_visual_feature_loss.item() / int(os.environ["NP"])
                        train_logit_loss = train_logit_loss.item() / int(os.environ["NP"])
                        visual_feature_losses = []
                        logit_losses = []

                    if int(os.environ["LOCAL_RANK"]) == 0:

                        time_minutes =(time.time() - s) / 60
                        self.writer.add_scalar("train/loss", train_loss, self.n_update)
                        self.writer.add_scalar("train/time", time_minutes, self.n_update)
                        if self.foundation_model_type == "video_llama2":
                            self.writer.add_scalar("train/visual_feature_loss", train_visual_feature_loss, self.n_update)
                            self.writer.add_scalar("train/logit_loss", train_logit_loss, self.n_update)

                        logger.info(f"---------- Update {self.n_update} ----------")
                        logger.info(f"train loss:{train_loss:.5f}")
                        if self.foundation_model_type == "video_llama2":
                            logger.info(f"visual feature loss: {train_visual_feature_loss:.5f}")
                            logger.info(f"logit loss: {train_logit_loss:.5f}")
                        logger.info(f"time:{time_minutes:.2f} [min]")
                        s = time.time()
                if self.n_update % self.save_interval == 0:
                    self.save_checkpoint(
                        file_name=f"ckpt.{save_cnt}.pth",
                        extra_state=None,
                    )
                    save_cnt += 1
                if self.n_update % self.val_interval == 0:
                    self.evaluate()
                    self.instruction_predictor.train()


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START iprl pretraining!\n")
    f.close()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    n_proc = int(os.environ["NP"])
    gpu_id = rank % world_size
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(backend="NCCL", init_method="env://", world_size=n_proc)
    print(f"rank: {rank}, world_size: {world_size}, gpu_id: {gpu_id}, n_proc: {n_proc}\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='the path to config.')
    parser.add_argument('--model-dir', help='the directory to save trained model.')
    parser.add_argument('--use-lmdb', help='whether to use lmdb dataset or not.')
    parser.add_argument('--log-interval', help='the interval to log about training.')
    parser.add_argument('--save-interval', help='the interval to save the trained model.')
    parser.add_argument('--val-interval', help='the interval to evaluate the trained model.')
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()
    
    os.makedirs(args.model_dir, exist_ok=True)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{args.model_dir}/train.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if rank == 0:
        logger.info("Start!")

    if "ss_baselines" in args.config:
        config = ss_get_config(args.config, args.opts, args.model_dir, 'train', False)
        environment_type = "ss1-savi"
    elif "habitat_baselines" in args.config:
        config = habitat_get_config(args.config, args.opts, args.model_dir)
        environment_type = "habitat-objnav"
    else:
        raise Exception(f"args.config: {args.config}")
    
    logger.info(config)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)

    trainer = OffPolicyEPRLPreTrainer(
        config=config, 
        logger=logger,
        gpu_id=gpu_id,
        model_dir=args.model_dir,
        use_lmdb_dataset=("True" == args.use_lmdb),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        val_interval=int(args.val_interval),
        environment_type=environment_type,
    )
    trainer.train()
