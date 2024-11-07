import os
import sys
import time
import logging
import argparse
import contextlib
import datetime


import yaml
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import all_reduce
from gym import spaces

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")

from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import IPRLPretrainingDataset, CollateFn, compute_spectrogram
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_lmdb_dataset import IPRLPretrainingLMDBDataset
from ss_baselines.saven.config.default import get_config

sys.path.append("/home/4/ud02274/navigation/myss")
from xgenerator.common.load_lmdb import PAD_IDX
from xgenerator.common.lang import tokens2sentences, R2RLang


def setup_instruction_predictor(
    iprl_cfg,
    ppo_cfg,
    smt_cfg,
    belief_cfg,
    device,
    observation_spaces,
    action_spaces,
    has_distractor_sound,
    pretrained,
    pretrained_weights,
):
    # circular import になるのでここで import
    from ss_baselines.saven.iprl_pretraining.common.instruction_predictor import AudioNavSMTInstructionPredictor

    instruction_predictor = AudioNavSMTInstructionPredictor(
        device=device,
        observation_space=observation_spaces,
        action_space=action_spaces,
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
        # for k, v in pretrained_state["state_dict"].items():
        #     print(f"{k}: {np.shape(v)}")
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

    instruction_predictor.to(device)
        
    return instruction_predictor, iprl_loss_fn

def save_checkpoint(
    instruction_predictor, config, file_name, extra_state=None
) -> None:
    checkpoint = {
        "state_dict": instruction_predictor.state_dict(),
        "config": config,
    }
    
    if extra_state is not None:
        checkpoint["extra_state"] = extra_state
    torch.save(
        checkpoint, os.path.join(config.CHECKPOINT_FOLDER, file_name)
    )


def rollout(instruction_predictor, loss_fn, inputs, targets, logger, visualize, force_student=False):
    if force_student:
        inputs["target"] = None
    else:
        inputs["target"] = targets # (batch, instr_len)

    logits = instruction_predictor(
        observations=inputs,
        prev_actions=inputs["action"],
        masks=None,
        ext_memory=None,
        ext_memory_masks=inputs["mask"],
    )
    targets = targets.permute(1, 0)[1:, :] # (instr_len-1, batch)
    if visualize and int(os.environ["RANK"]) == 0:
        lang = R2RLang("r2r")
        _, iprl_tokens = logits.max(2)
        pred_sentence = tokens2sentences(iprl_tokens, lang)[0]
        true_sentence = tokens2sentences(targets, lang)[0]
        logger.info(f"Pred: {pred_sentence}")
        logger.info(f"True: {true_sentence}")
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    return loss


def evaluate(instruction_predictor, val_dataloader, loss_fn, logger, writer, n_update):
    if int(os.environ["RANK"]) == 0:
        logger.info(f"========== Evaluation ==========")
    instruction_predictor.eval()
    s = time.time()
    losses = []
    for i, (inputs, targets) in enumerate(val_dataloader):
        loss = rollout(instruction_predictor, loss_fn, inputs, targets, logger, True, True)
        losses.append(loss.item())
    loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
    all_reduce(loss)
    if int(os.environ["RANK"]) == 0:
        loss = loss.item() / int(os.environ["NP"])
        writer.add_scalar("eval/loss", loss, n_update)
        logger.info(f"Loss: {loss:.5f}")
        logger.info(f"Time: {(time.time() - s)/60:.2f} [min]")


def train_one_step(
    instruction_predictor,
    loss_fn,
    inputs,
    targets,
    logger,
    visualize,
):
    instruction_predictor.train()
    instruction_predictor.optimizer.zero_grad()
    loss = rollout(instruction_predictor, loss_fn, inputs, targets, logger, visualize)
    loss.backward()
    instruction_predictor.optimizer.step()
    return loss


def train(
    config,
    logger,
    gpu_id,
    model_dir,
    use_lmdb_dataset,
    log_interval,
    save_interval,
    val_interval,
):
    logger.info(f"device: {torch.device('cuda', gpu_id)}")

    spectrogram_shape = compute_spectrogram(np.ones((2, config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE))).shape
    observation_space = spaces.Dict({
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
    action_space = spaces.Discrete(4)
    
    instruction_predictor, loss_fn = setup_instruction_predictor(
        iprl_cfg=config.RL.PPO.INSTRUCTION_PREDICTOR,
        ppo_cfg=config.RL.PPO,
        smt_cfg=config.RL.PPO.SCENE_MEMORY_TRANSFORMER,
        belief_cfg=config.RL.PPO.BELIEF_PREDICTOR,
        device=torch.device("cuda", gpu_id),
        observation_spaces=observation_space,
        action_spaces=action_space,
        has_distractor_sound=config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND,
        pretrained=config.RL.DDPPO.pretrained,
        pretrained_weights=config.RL.DDPPO.pretrained_weights,
    )
    instruction_predictor.train()
    
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    if int(os.environ["RANK"]) == 0:
        logger.info("LOADING DATA...")
    
    if use_lmdb_dataset:
        train_split = config.TASK_CONFIG.DATASET.SPLIT
        train_dataset = IPRLPretrainingLMDBDataset(
            # config.TASK_CONFIG, train_split, "./data/lmdb_dataset/iprl_pretrain/train_cr02", 100398,
            config.TASK_CONFIG, train_split, "./data/lmdb_dataset/iprl_pretrain/train", 502103,
            environment_type="ss1-savi",
            device=torch.device("cuda", gpu_id),
        )
        if "past" in train_split:
            val_split = "val_w_past_instruction"
        else:
            val_split = "val_w_instruction"
        val_dataset = IPRLPretrainingLMDBDataset(
            config.TASK_CONFIG, val_split, "./data/lmdb_dataset/iprl_pretrain/val", 500,
            environment_type="ss1-savi",
            device=torch.device("cuda", gpu_id),
        )
    else:
        # train_dataset = IPRLPretrainingDataset(config.TASK_CONFIG, config.SENSORS)
        raise NotImplementedError("use_lmdb_dataset must be True.") # val_datasetに対応させてないので

    my_collate_fn = CollateFn(
        use_foundation_model=config.FOUNDATION_MODEL.model_type == "video_llama2",
        max_instr_len=config.FOUNDATION_MODEL.max_instr_len,
        environment_type="ss1-savi",
    )
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=True,
        rank=int(os.environ["RANK"]),
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=False,
        rank=int(os.environ["RANK"]),
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.RL.PPO.num_steps,
        drop_last=True,
        collate_fn=my_collate_fn,
        sampler=train_sampler,
        # num_workers=2,
    )
    val_dataloader = DataLoader(
        val_dataset,
        # num_workers=2,
        batch_size=config.RL.PPO.num_steps,
        drop_last=False,
        collate_fn=my_collate_fn,
        sampler=val_sampler,
    )
    n_batch = len(train_dataloader)
    if int(os.environ["RANK"]) == 0:
        logger.info(f"The number of train data: {len(train_dataset)}, batch: {n_batch}")
        logger.info("FINISH LOADING DATA!")
        logger.info("START TRAINING...")

    tb_log_dir = f"{model_dir}/tb"
    os.makedirs(tb_log_dir, exist_ok=True)

    s = time.time()
    with (
            SummaryWriter(log_dir=tb_log_dir)
            if int(os.environ["RANK"]) == 0
            else contextlib.suppress()
    ) as writer:
        n_loop = 0
        save_cnt = 0
        losses = []
        while True:
            n_loop += 1
            for j, (inputs, targets) in enumerate(train_dataloader):
                n_update = n_batch*(n_loop-1) + j
                loss = train_one_step(
                    instruction_predictor,
                    loss_fn,
                    inputs,
                    targets,
                    logger,
                    False,
                )
                losses.append(loss.item())

                if n_update % log_interval == 0:
                    train_loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
                    all_reduce(train_loss)
                    train_loss = train_loss.item() / int(os.environ["NP"])
                    losses = []

                    if int(os.environ["RANK"]) == 0:
                        
                        time_minutes =(time.time() - s) / 60
                        writer.add_scalar("train/loss", train_loss, n_update)
                        writer.add_scalar("train/time", time_minutes, n_update)

                        logger.info(f"---------- Update {n_update} ----------")
                        logger.info(f"train loss:{train_loss:.5f}")
                        logger.info(f"time:{time_minutes:.2f} [min]")
                        s = time.time()
                if n_update % save_interval == 0:
                    save_checkpoint(
                        instruction_predictor,
                        config=config,
                        file_name=f"ckpt.{save_cnt}.pth",
                        extra_state=None,
                    )
                    save_cnt += 1
                if n_update % val_interval == 0:
                    evaluate(
                        instruction_predictor,
                        val_dataloader,
                        loss_fn,
                        logger,
                        writer,
                        n_update,
                    )


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START iprl pretraining!\n")
    f.close()
    WORLD_SIZE = int(os.getenv("NP"))
    NNODES = int(os.getenv("NNODES"))
    NPERNODE = int(os.getenv("NPERNODE"))
    rank = int(os.getenv("OMPI_COMM_WORLD_RANK", "0"))
    os.environ["NODE_RANK"]=str(rank//NPERNODE)
    os.environ["LOCAL_RANK"]=str(rank%NPERNODE)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
    os.environ["NNODES"] = str(NNODES)

    rank = int(os.environ["RANK"])
    world_size = torch.cuda.device_count()
    n_proc = int(os.environ["NP"])
    gpu_id = rank % NPERNODE
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(
        backend="NCCL", # "GLOO",
        init_method="env://",
        world_size=n_proc,
    )
    print(f"rank: {rank}, world_size: {world_size}, gpu_id: {gpu_id}, n_proc: {n_proc}\n")
    print(f"rank: {os.environ['RANK']}, world_size: {os.environ['WORLD_SIZE']}, LOCAL_RANK: {os.environ['LOCAL_RANK']}")
    print(f"torch.distributed.get_rank(): {torch.distributed.get_rank()}")

    
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
    if int(os.environ['RANK']) == 0:
        logger.info("Start!")
    
    config = get_config(args.config, args.opts, args.model_dir, 'train', False)
    logger.info(config)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    
    train(
        config=config, 
        logger=logger,
        gpu_id=gpu_id,
        model_dir=args.model_dir,
        use_lmdb_dataset=("True" == args.use_lmdb),
        log_interval=int(args.log_interval),
        save_interval=int(args.save_interval),
        val_interval=int(args.val_interval),
    )
