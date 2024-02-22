import os
import sys
import time
import logging
import argparse

import lmdb
import lmdb
import msgpack_numpy
import numpy as np
import torch
from gym import spaces

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")

from habitat.datasets import make_dataset
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import compute_spectrogram
from ss_baselines.saven.config.default import get_config
from soundspaces.utils import generate_video
from off_policy_train import setup_instruction_predictor
from xgenerator.common.lang import tokens2sentences, R2RLang


def get_data(index, episode, dataset_path, gen_video, model_dir):
    env = lmdb.open(dataset_path, readonly=True, lock=False)
    with env.begin() as txn:
        value = txn.get(str(index).encode('latin-1'))
        value = msgpack_numpy.unpackb(value, object_hook=msgpack_numpy.decode)
    env.close()
    image_seq = value[0]   # (seq_len, 1, image_shape)
    audio_seq = value[1]   # (seq_len, 1, spectrogram_shape)
    pose_seq = value[2]    # (seq_len, 1, 4)
    action_seq = value[3]  # (seq_len, 1)
    category = value[4]    # (21,)
    location = value[5]    # (2,)

    step = pose_seq[-1, 0, 3]
    instruction = np.array(episode.instructions)[:, int(step)]

    if gen_video:
        os.makedirs(f"{model_dir}/offpolicy_eval_video", exist_ok=True)
        generate_video(torch.from_numpy(image_seq.copy()), f"{model_dir}/offpolicy_eval_video/image_seq_{index}.mp4")

    seq_len = np.shape(image_seq)[0]
    data = {
        "rgb": torch.from_numpy(image_seq[:, :, :, :, :3]).cuda().float(),
        "depth": torch.from_numpy(image_seq[:, :, :, :, 3:4]).cuda().float(),
        "pose": torch.from_numpy(pose_seq).cuda().float(),
        "spectrogram": torch.from_numpy(audio_seq).cuda().float(),
        "action": torch.from_numpy(action_seq).view(seq_len, 1, 1).cuda().float(),
        "category": torch.from_numpy(category).view(1, 21).cuda().float(),
        "location": torch.from_numpy(location).view(1, 2).cuda().float(),
        "target": torch.from_numpy(np.array(instruction)).view(1, 40).cuda(),
        "mask": None,
        "seq_lengths": torch.from_numpy(np.array([seq_len])).cuda(),
    }
    return data


def eval(
    config,
    gpu_id,
    logger,
    model_dir,
    indices,
    dataset_path,
):
    logger.info(f"device: {torch.device('cuda', gpu_id)}")

    lang = R2RLang("r2r")
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    episodes = dataset.episodes

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

    # state_dict = torch.load(config.RL.DDPPO.pretrained_weights, map_location="cpu")["state_dict"]
    # state_dict_removed = {}
    # for k, v in state_dict.items():
    #     if not "actor_critic.action_distribution." in k and not "actor_critic.critic." in k:
    #         state_dict_removed[k[len("actor_critic.net."):]] = v       
    # instruction_predictor.load_state_dict(
    #     state_dict_removed
    # )
    instruction_predictor.load_state_dict(
        torch.load(config.RL.DDPPO.pretrained_weights, map_location="cpu")["state_dict"]
    )
    instruction_predictor.eval()

    for index in indices:
        print(f"index: {index}")
        data = get_data(
            index=index,
            episode=episodes[index],
            dataset_path=dataset_path,
            gen_video=True,
            model_dir=model_dir,
        )
        logits = instruction_predictor(
            observations=data,
            prev_actions=data["action"],
            masks=None,
            ext_memory=None,
            ext_memory_masks=None,
        )
        targets = data["target"].permute(1, 0)[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


        _, iprl_tokens = logits.max(2)
        pred_sentence = tokens2sentences(iprl_tokens, lang)[0]
        true_sentence = tokens2sentences(data["target"].permute(1, 0), lang)[0]

        logger.info(f"-------- index: {index} --------")
        logger.info(f"seq_len: {data['action'].size()[0]}")
        logger.info(f"loss: {loss}")
        logger.info(f"Pred: {pred_sentence}")
        logger.info(f"True: {true_sentence}")


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START iprl pretraining off-policy eval!\n")
    f.close()

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='the path to config.')
    parser.add_argument('--model-dir', help='the directory of model to eval.')
    parser.add_argument('--indices', type=int, nargs="*", help='data indices for eval.')
    parser.add_argument('--past-or-future', type=str, help='the model to predict the past or future instruction.')
    parser.add_argument('--dataset-type', type=str, help='the type of dataset') # train, eval, test
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    args.opts.append("TASK_CONFIG.DATASET.SPLIT")
    if args.past_or_future == "past":
        split = f"{args.dataset_type}_w_past_instruction"
    elif args.past_or_future == "future":
        split = f"{args.dataset_type}_w_instruction"
    else:
        raise Exception(f"args.past_or_future: {args.past_or_future}")
    print(args.past_or_future, split)
    args.opts.append(split)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{args.model_dir}/eval.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start!")
    
    config = get_config(args.config, args.opts, args.model_dir, 'eval', False)
    logger.info(config)
    os.makedirs(config.CHECKPOINT_FOLDER, exist_ok=True)
    
    eval(
        config=config, 
        gpu_id=0,
        logger=logger,
        model_dir=args.model_dir,
        indices=args.indices,
        dataset_path=f"./data/lmdb_dataset/iprl_pretrain_{args.dataset_type}"
    )






