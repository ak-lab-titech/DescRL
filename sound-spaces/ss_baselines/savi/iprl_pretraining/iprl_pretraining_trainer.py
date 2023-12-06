import contextlib
import os
import sys
import random
import time
import glob
from typing import Dict, List, Any
from collections import defaultdict, deque

import numpy as np
import torch
import torch.distributed as distrib
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.savi.iprl_pretraining.iprl_pretraining_rollout_storage import IPRLPretrainingRolloutStorage
from ss_baselines.savi.iprl_pretraining.instruction_predictor import AudioNavSMTInstructionPredictorDDP
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import batch_obs, linear_decay
from ss_baselines.savi.ddppo.algo.ddp_utils import (
    EXIT,
    REQUEUE,
    add_signal_handlers,
    init_distrib_slurm,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from ss_baselines.savi.ddppo.algo.ddppo import DDPPO
from ss_baselines.savi.models.belief_predictor import BeliefPredictor, BeliefPredictorDDP
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor
from soundspaces.tasks.nav import LocationBelief, CategoryBelief, SpectrogramSensor

sys.path.append("/home/0/19B30511/av-nav/myss")
from xgenerator.common.load_lmdb import PAD_IDX


class IPRLPretrainingTrainer(BaseRLTrainer):
    SHORT_ROLLOUT_THRESHOLD: float = 0.25

    def __init__(self, config=None):
        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            config = interrupted_state["config"]
        super().__init__(config)
    
    def _setup_instruction_predictor(self, ppo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)

        action_space = self.envs.action_spaces[0]
        self.action_space = action_space

        has_distractor_sound = self.config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND

        smt_cfg = ppo_cfg.SCENE_MEMORY_TRANSFORMER
        belief_cfg = ppo_cfg.BELIEF_PREDICTOR
        iprl_cfg = ppo_cfg.INSTRUCTION_PREDICTOR

        self.instruction_predictor = AudioNavSMTInstructionPredictorDDP(
            device=self.device,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            direct_map_size=self.config.TASK_CONFIG.SIMULATOR.DIRECT_MAP_SIZE,
            goal_num=self.config.TASK_CONFIG.SIMULATOR.AUDIO.NUM,
            iprl_max_instr_len=iprl_cfg.max_instr_len,
            iprl_num_decoder_layers=iprl_cfg.num_decoder_layers,
            iprl_vocab_emb_size=iprl_cfg.vocab_emb_size,
            iprl_emb_size=iprl_cfg.emb_size,
            iprl_nhead=iprl_cfg.nhead,
            iprl_dim_feedforward=iprl_cfg.dim_feedforward,
            iprl_dropout=iprl_cfg.dropout,
            iprl_use_gt_D=iprl_cfg.iprl_use_gt_D,
            iprl_feedback=iprl_cfg.feedback,
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
        )
        self.instruction_predictor.optimizer = torch.optim.Adam(
            self.instruction_predictor.parameters(),
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
        )
        self.iprl_loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        if smt_cfg.freeze_encoders:
            self._static_smt_encoder = True
            self.instruction_predictor.freeze_encoders()

        if ppo_cfg.use_belief_predictor:
            smt = self.instruction_predictor.smt_state_encoder
            bp_class = BeliefPredictorDDP if belief_cfg.online_training else BeliefPredictor
            self.belief_predictor = bp_class(belief_cfg, self.device, smt._input_size, smt._pose_indices, self.config.TASK_CONFIG.SIMULATOR.AUDIO.NUM,
                                             smt.hidden_state_size, self.envs.num_envs, has_distractor_sound
                                             ).to(device=self.device)
            if belief_cfg.online_training:
                params = list(self.belief_predictor.predictor.parameters())
                if belief_cfg.train_encoder:
                    params += list(self.instruction_predictor.goal_encoder.parameters()) + \
                              list(self.instruction_predictor.visual_encoder.parameters()) + \
                              list(self.instruction_predictor.action_encoder.parameters())
                self.belief_predictor.optimizer = torch.optim.Adam(params, lr=belief_cfg.lr)
            self.belief_predictor.freeze_encoders() # classifierをfreezeさせる。online_trainingでなければpredictorも
        self.instruction_predictor.to(self.device)

        if self.config.RL.DDPPO.pretrained:
            # load weights for both actor critic and the encoder
            pretrained_state = torch.load(self.config.RL.DDPPO.pretrained_weights, map_location="cpu")
            self.instruction_predictor.load_state_dict(
                {
                    k[len("actor_critic."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder" not in k and
                       "actor_critic.net.smt_state_encoder" not in k
                },
                strict=False
            )
            self.instruction_predictor.visual_encoder.rgb_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.rgb_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.rgb_encoder." in k
                },
            )
            self.instruction_predictor.visual_encoder.depth_encoder.load_state_dict(
                {
                    k[len("actor_critic.net.visual_encoder.depth_encoder."):]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if "actor_critic.net.visual_encoder.depth_encoder." in k
                },
            )
            self.belief_predictor.predictor.load_state_dict(
                {
                    k[len("predictor."):]: v
                    for k, v in pretrained_state['belief_predictor'].items()
                    if "predictor." in k
                }
            )
        
        self.ppo_epoch = ppo_cfg.ppo_epoch
        self.num_mini_batch = ppo_cfg.num_mini_batch
    
    def _collect_rollout_step(
            self, rollouts
        ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()

        step_observation = {
            k: v[rollouts.step] for k, v in rollouts.observations.items()
        }
        external_memory_features = self.instruction_predictor.get_features(
            step_observation,
            rollouts.prev_actions[rollouts.step],
        )

        actions = step_observation["oracle_action_sensor"]

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([int(a[0].item()) for a in actions])
        observations, _, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=self.device
        )

        rollouts.insert(
            batch,
            actions,
            masks.to(device=self.device),
            external_memory_features,
            dones,
        )

        if self.config.RL.PPO.use_belief_predictor:
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            self.belief_predictor.update(step_observation, dones)
            if self.config.RL.PPO.BELIEF_PREDICTOR.use_label_belief:
                rollouts.observations[CategoryBelief.cls_uuid][rollouts.step].copy_(
                    step_observation[CategoryBelief.cls_uuid]
                )
            if self.config.RL.PPO.BELIEF_PREDICTOR.use_location_belief:
                rollouts.observations[LocationBelief.cls_uuid][rollouts.step].copy_(
                    step_observation[LocationBelief.cls_uuid]
                )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def train(self) -> None:
        self.local_rank, tcp_store = init_distrib_slurm(
            self.config.RL.DDPPO.distrib_backend
        )
        add_signal_handlers()

        # Stores the number of workers that have finished their rollout
        num_rollouts_done_store = distrib.PrefixStore(
            "rollout_tracker", tcp_store
        )
        num_rollouts_done_store.set("num_done", "0")

        self.world_rank = distrib.get_rank()
        self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        # Multiply by the number of simulators to make sure they also get unique seeds
        self.config.TASK_CONFIG.SEED += (
            self.world_rank * self.config.NUM_PROCESSES
        )
        self.config.freeze()

        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        if (
            not os.path.isdir(self.config.CHECKPOINT_FOLDER)
            and self.world_rank == 0
        ):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_instruction_predictor(ppo_cfg)
        self.instruction_predictor.init_distributed(find_unused_params=True)
        if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
            self.belief_predictor.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "instruction_predictor number of trainable parameters: {}".format(
                    sum(
                        param.numel()
                        for param in self.instruction_predictor.parameters()
                        if param.requires_grad
                    )
                )
            )
            if ppo_cfg.use_belief_predictor:
                logger.info(
                    "belief predictor number of trainable parameters: {}".format(
                        sum(
                            param.numel()
                            for param in self.belief_predictor.parameters()
                            if param.requires_grad
                        )
                    )
                )
            logger.info(f"config: {self.config}")

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        obs_space = self.envs.observation_spaces[0]
        if ppo_cfg.use_external_memory:
            memory_dim = self.instruction_predictor.memory_dim
        else:
            memory_dim = None

        rollouts = IPRLPretrainingRolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.action_space,
            ppo_cfg.use_external_memory,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size + ppo_cfg.num_steps,
            ppo_cfg.SCENE_MEMORY_TRANSFORMER.memory_size,
            memory_dim,
        )
        rollouts.to(self.device)

        if self.config.RL.PPO.use_belief_predictor:
            self.belief_predictor.update(batch, None)

        for sensor in rollouts.observations:
            if sensor == "depth":
                batch_sensor = np.squeeze(batch[sensor], axis=4)
            elif sensor == "generated_instruction":
                batch_sensor = np.squeeze(batch[sensor], axis=1)
            elif sensor == "oracle_action_sensor":
                batch_sensor = batch[sensor].reshape(-1, 1)
            else:
                batch_sensor = batch[sensor]
            rollouts.observations[sensor][0].copy_(batch_sensor)

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.instruction_predictor.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        # Try to resume at previous checkpoint (independent of interrupted states)
        count_steps_start, count_checkpoints, start_update = self.try_to_resume_checkpoint()
        count_steps = count_steps_start

        interrupted_state = load_interrupted_state()
        if interrupted_state is not None:
            self.instruction_predictor.load_state_dict(interrupted_state["state_dict"])
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(interrupted_state["belief_predictor"])
            self.instruction_predictor.optimizer.load_state_dict(
                interrupted_state["optim_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_sched_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with (
            TensorboardWriter(
                self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
            )
            if self.world_rank == 0
            else contextlib.suppress()
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set() and self.world_rank == 0:
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        state_dict = dict(
                                state_dict=self.instruction_predictor.state_dict(),
                                optim_state=self.instruction_predictor.optimizer.state_dict(),
                                lr_sched_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            )
                        if self.config.RL.PPO.use_belief_predictor:
                            state_dict['belief_predictor'] = self.belief_predictor.state_dict()
                        save_interrupted_state(state_dict)

                    requeue_job()
                    return

                count_steps_delta = 0
                self.instruction_predictor.eval()
                if self.config.RL.PPO.use_belief_predictor:
                    self.belief_predictor.eval()
                for step in range(ppo_cfg.num_steps):

                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps_delta += delta_steps

                    # This is where the preemption of workers happens.  If a
                    # worker detects it will be a straggler, it preempts itself!
                    if (
                        step
                        >= ppo_cfg.num_steps * self.SHORT_ROLLOUT_THRESHOLD
                    ) and int(num_rollouts_done_store.get("num_done")) > (
                        self.config.RL.DDPPO.sync_frac * self.world_size
                    ):
                        break

                num_rollouts_done_store.add("num_done", 1)

                self.instruction_predictor.train()
                if self.config.RL.PPO.use_belief_predictor:
                    self.belief_predictor.train()
                    self.belief_predictor.set_eval_encoders()
                if self._static_smt_encoder:
                    self.instruction_predictor.set_eval_encoders()

                if ppo_cfg.use_belief_predictor and ppo_cfg.BELIEF_PREDICTOR.online_training:
                    location_predictor_loss, prediction_accuracy = self.train_belief_predictor(rollouts)
                else:
                    location_predictor_loss = 0
                    prediction_accuracy = 0
                (
                    delta_pth_time,
                    iprl_loss
                ) = self.train_instruction_predictor(rollouts)
                pth_time += delta_pth_time

                stats = torch.tensor(
                    [location_predictor_loss, prediction_accuracy, count_steps_delta, iprl_loss],
                    device=self.device,
                )
                distrib.all_reduce(stats)
                count_steps += stats[2].item()

                if self.world_rank == 0:
                    num_rollouts_done_store.set("num_done", "0")

                    losses = [
                        stats[0].item() / self.world_size,
                        stats[1].item() / self.world_size,
                        stats[3].item() / self.world_size,
                    ]

                    writer.add_scalar("Policy/predictor_loss", losses[0], count_steps)
                    writer.add_scalar("Policy/predictor_accuracy", losses[1], count_steps)
                    writer.add_scalar("Policy/iprl_loss", losses[2], count_steps)
                    writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                    # log stats
                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}\t".format(
                                update,
                                (count_steps - count_steps_start)
                                / ((time.time() - t_start) + prev_time),
                            )
                        )

                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                            "frames: {}".format(
                                update, env_time, pth_time, count_steps
                            )
                        )
                        logger.info("predictor_loss: {:.5f}, predictor_accuracy: {:.5f}, iprl_loss: {:.5f}".format(
                            losses[0], losses[1], losses[2]
                        ))

                    # checkpoint model
                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            dict(step=count_steps),
                        )
                        count_checkpoints += 1

            self.envs.close()

    @staticmethod
    def search_dict(ckpt_dict, encoder_name):
        encoder_dict = {}
        for key, value in ckpt_dict['state_dict'].items():
            if encoder_name in key:
                encoder_dict['.'.join(key.split('.')[3:])] = value

        return encoder_dict

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.instruction_predictor.state_dict(),
            "config": self.config,
        }
        if self.config.RL.PPO.use_belief_predictor:
            checkpoint["belief_predictor"] = self.belief_predictor.state_dict()
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            self.instruction_predictor.load_state_dict(ckpt_dict["state_dict"])
            if self.config.RL.PPO.use_belief_predictor:
                self.belief_predictor.load_state_dict(ckpt_dict["belief_predictor"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def train_belief_predictor(self, rollouts):
        bp = self.belief_predictor
        num_epoch = 5
        num_mini_batch = 1

        value_loss_epoch = 0
        running_regressor_corrects = 0
        num_sample = 0

        for e in range(num_epoch):
            data_generator = rollouts.recurrent_generator(
                num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    _,
                    _,
                    _,
                    _,
                    _,
                ) = sample

                bp.optimizer.zero_grad()

                inputs = obs_batch[SpectrogramSensor.cls_uuid].permute(0, 3, 1, 2)
                preds = bp.cnn_forward(obs_batch)

                masks = (torch.sum(torch.reshape(obs_batch[SpectrogramSensor.cls_uuid],
                        (obs_batch[SpectrogramSensor.cls_uuid].shape[0], -1)), dim=1, keepdim=True) != 0).float()
                gts = obs_batch[IntegratedPointGoalGPSAndCompassSensor.cls_uuid]
                transformed_gts = []
                for gn in range(self.config.TASK_CONFIG.SIMULATOR.AUDIO.NUM):
                    transformed_gts = transformed_gts + [gts[:, 2*gn + 1], -gts[:, 2*gn]]
                transformed_gts = torch.stack(transformed_gts, dim=1)
                masked_preds = masks.expand_as(preds) * preds
                masked_gts = masks.expand_as(transformed_gts) * transformed_gts
                loss = bp.regressor_criterion(masked_preds, masked_gts)

                bp.before_backward(loss)
                loss.backward()
                # self.after_backward(loss)

                bp.optimizer.step()
                value_loss_epoch += loss.item()

                rounded_preds = torch.round(preds)
                bitwise_close = torch.bitwise_and(torch.isclose(rounded_preds[:, 0], transformed_gts[:, 0]),
                                                  torch.isclose(rounded_preds[:, 1], transformed_gts[:, 1]))
                running_regressor_corrects += torch.sum(torch.bitwise_and(bitwise_close, masks.bool().squeeze(1)))
                num_sample += torch.sum(masks).item()

        value_loss_epoch /= num_epoch * num_mini_batch
        if num_sample == 0:
            prediction_accuracy = 0
        else:
            prediction_accuracy = running_regressor_corrects / num_sample

        return value_loss_epoch, prediction_accuracy
    
    def train_instruction_predictor(self, rollouts):
        t_update_model = time.time()

        iprl_loss_epoch = 0
        for e in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(
                self.num_mini_batch
            )

            for sample in data_generator:
                (
                    obs_batch,
                    actions_batch,
                    prev_actions_batch,
                    masks_batch,
                    external_memory,
                    external_memory_masks,
                ) = sample

                iprl_logits = self.instruction_predictor(
                    obs_batch,
                    prev_actions_batch,
                    masks_batch,
                    external_memory,
                    external_memory_masks,
                )

                iprl_targets = obs_batch["generated_instruction"].permute(1, 0)[1:, :].long() # (instr_len, batch)
                iprl_loss = self.iprl_loss_fn(iprl_logits.view(-1, iprl_logits.shape[-1]), iprl_targets.reshape(-1))

                self.instruction_predictor.optimizer.zero_grad()

                total_loss = iprl_loss

                self.instruction_predictor.before_backward(total_loss)
                total_loss.backward()

                self.instruction_predictor.optimizer.step()

                iprl_loss_epoch += iprl_loss.item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        iprl_loss_epoch /= num_updates

        rollouts.after_update()
        return (
            time.time() - t_update_model,
            iprl_loss_epoch,
        )
