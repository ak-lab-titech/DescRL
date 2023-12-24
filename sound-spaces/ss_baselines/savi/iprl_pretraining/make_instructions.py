import sys
import argparse
import gzip
import json
import os
import time

import yaml
import numpy as np
import torch

sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")
sys.path.append("/home/0/19B30511/av-nav/myss")

from habitat.sims import make_sim
from habitat.datasets import make_dataset
from habitat_sim.utils.common import quat_from_angle_axis
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config
from ss_baselines.savi.config.default import get_config
from xgenerator.common.lang import R2RLang
from common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX
from xgenerator.transformer_speaker.model import Seq2SeqTransformer


def setup_instruction_generator(
    xgenerator_path,
    ckpt_num,
    ):
    lang = R2RLang(name="r2r_train")
    vocab_size = lang.vocab_size

    # xgenerator_path = config["GENERATED_INSTRUCTION"]["XGENERATOR_PATH"]
    # ckpt_num = config["GENERATED_INSTRUCTION"]["XGENERATOR_CKPT"]
    # max_instr_len = config["GENERATED_INSTRUCTION"]["MAX_INSTRUCTION_LENGTH"]
    # future_step_num = config["GENERATED_INSTRUCTION"]["FUTURE_STEP_NUM"]

    with open(f"../{xgenerator_path}/config.yaml", "r") as yml:
        xgenerator_config = yaml.safe_load(yml)

    instruction_generator = Seq2SeqTransformer(
        num_encoder_layers=xgenerator_config["model"]["num_encoder_layers"],   
        num_decoder_layers=xgenerator_config["model"]["num_decoder_layers"],
        emb_size=xgenerator_config["model"]["emb_size"],
        vocab_emb_size=xgenerator_config["model"]["vocab_embedding_size"],
        use_semantic=xgenerator_config["model"]["use_semantic"],
        nhead=xgenerator_config["model"]["nhead"],
        use_image_feature=xgenerator_config["train"]["use_image_feature"],
        vocab_size=vocab_size,
        glove=lang.glove_vec,
        dim_feedforward=xgenerator_config["model"]["dim_feedforward"],
        dropout=xgenerator_config["model"]["dropout_ratio"],
    )
    instruction_generator.load_state_dict(
        torch.load(
            f"../{xgenerator_path}/data/{ckpt_num}/seq2seq.pth",
            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            )
    )
    if torch.cuda.is_available():
        instruction_generator = instruction_generator.to(torch.device("cuda"))
    return instruction_generator


def get_obs_seqs(sim, future_step_num):
    current_previous_step_collided = sim._previous_step_collided
    current_is_episode_active = sim._is_episode_active
    current_receiver_position_index = sim._receiver_position_index
    current_rotation_angle = sim._rotation_angle
    current_episode_step_count = sim._episode_step_count
    current_prev_sim_obs = sim._prev_sim_obs

    image_seqs_list = []
    action_seqs_list = []
    oracle_actions = sim.get_oracle_actions_from_current_pos()
    future_step_cnt = 0
    # for文だと-1の時に対応できないのでwhile
    while True:
        future_step_cnt += 1
        sim_obs = sim._get_sim_observation()
        observations = sim._sensor_suite.get_observations(sim_obs)
        image_shape = np.shape(observations["depth"])
        
        if len(image_shape) == 4:
            depth_img = np.squeeze(observations["depth"], axis=3)
        else:
            depth_img = observations["depth"]
        rgb_img = observations["rgb"] / 255.0
            
        if "semantic" in observations.keys():
            semantic_img = np.squeeze(observations["semantic"], axis=3) / 255.0
            image = np.concatenate([rgb_img, depth_img, semantic_img], 2).astype(np.float32)
        else:
            image = np.concatenate([rgb_img, depth_img], 2).astype(np.float32)

        action = oracle_actions[future_step_cnt-1]
        action_onehot = np.eye(4)[action].astype(np.int8)
        image_seqs_list.append([image])
        action_seqs_list.append([action_onehot])
        if action == 0 or future_step_cnt == future_step_num:
            break
        sim.step(action)
        
    sim._previous_step_collided = current_previous_step_collided
    sim._is_episode_active = current_is_episode_active
    sim._receiver_position_index = current_receiver_position_index
    sim._rotation_angle = current_rotation_angle
    sim._episode_step_count = current_episode_step_count
    sim._prev_sim_obs = current_prev_sim_obs
    sim.set_agent_state(
        position=list(sim.graph.nodes[sim._receiver_position_index]['point']),
        rotation=quat_from_angle_axis(np.deg2rad(sim._rotation_angle), np.array([0, 1, 0])),
    )

    image_seqs = np.array(image_seqs_list)
    action_seqs = np.array(action_seqs_list)
    
    return image_seqs, action_seqs


def make_batch(image_seqs, action_seqs, input_future_step_num):
    batch_size = np.shape(image_seqs)[0]
    batched_image_seqs = np.zeros(
        (input_future_step_num, batch_size) + np.shape(image_seqs[0][0]),
        np.float32,
    )
    batched_action_seqs = np.zeros(
        (input_future_step_num, batch_size, 4),
        np.float32,
    )
    path_masks = np.full((batch_size, input_future_step_num), True)

    for i in range(batch_size):
        image_seq = image_seqs[i:i+input_future_step_num, 0]
        seq_len = np.shape(image_seq)[0]
        action_seq = action_seqs[i:i+input_future_step_num, 0]
        batched_image_seqs[:seq_len, i] = image_seq
        batched_action_seqs[:seq_len, i] = action_seq
        path_masks[i, :seq_len] = False

    batched_image_seqs = torch.from_numpy(np.array(batched_image_seqs))
    batched_action_seqs = torch.from_numpy(np.array(batched_action_seqs))
    path_masks = torch.from_numpy(np.array(path_masks))
    if torch.cuda.is_available():
        batched_image_seqs = batched_image_seqs.cuda()
        batched_action_seqs = batched_action_seqs.cuda()
        path_masks = path_masks.cuda()

    return batched_image_seqs, batched_action_seqs, path_masks


def generate_instruction(
    instruction_generator,
    image_seqs,
    action_seqs,
    path_mask,
    max_instr_len,
):
    """
    image_seqs: (seq_l, batch, image_shape)
    action_seqs: (seq_l, batch, 4)
    """
    batch_size = np.shape(image_seqs)[1]
    ended = torch.full((1, batch_size), False)
    with torch.no_grad():
        memory = instruction_generator.encode(
            src_image=image_seqs,
            src_action=action_seqs,
            src_mask=None,
            src_padding_mask=path_mask,
        )
        past_tokens = torch.full((1, batch_size), BOS_IDX)
        if torch.cuda.is_available():
            past_tokens = past_tokens.cuda()
        
        for _ in range(max_instr_len-1):
            logits = instruction_generator.decode(
                trg=past_tokens,
                memory=memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=path_mask,
            )[-1, :, :] # (batch, vocab_size)
            _, next_tokens = logits.max(1)
            next_tokens = next_tokens.view(1, batch_size)
            next_tokens[ended] = PAD_IDX
            ended[next_tokens == EOS_IDX] = True
            past_tokens = torch.cat([past_tokens, next_tokens], dim=0)
    return past_tokens


def main(
    config,
    sensor_list,
    content_scenes_path,
    save_dataset_path,
):
    dataset = make_dataset(
        id_dataset=config.DATASET.TYPE,
        config=config.DATASET,
    )
    episodes = dataset.episodes
    sim_cfg = config.SIMULATOR
    sim_cfg.defrost()
    sim_cfg.SCENE_DATASET = episodes[0].scene_dataset_config
    sim_cfg.AGENT_0.SENSORS = sensor_list
    sim_cfg.SCENE = episodes[0].scene_id
    sim_cfg.freeze()
    sim = make_sim(
        sim_cfg.TYPE, config=sim_cfg,
    )
    instruction_predictor = setup_instruction_generator(
        config.TASK.GENERATED_INSTRUCTION.XGENERATOR_PATH,
        config.TASK.GENERATED_INSTRUCTION.XGENERATOR_CKPT,
    )

    scene_file_names = [
        f for f in os.listdir(content_scenes_path) if os.path.isfile(os.path.join(content_scenes_path, f))
    ]
    dict_dataset = {f: {'episodes': [], 'scene': f.split('/')[0]} for f in scene_file_names}

    print(f"length of episodes: {len(episodes)}\n")
    cnt = 0
    for episode in episodes:
        s = time.time()

        sim_cfg = merge_sim_episode_config(sim_cfg, episode)
        sim.reconfigure(sim_cfg)
        _ = sim.reset()
        image_seqs, action_seqs = get_obs_seqs(sim, -1)
        batched_image_seqs, batched_action_seqs, path_masks = make_batch(image_seqs, action_seqs, config.TASK.GENERATED_INSTRUCTION.FUTURE_STEP_NUM)

        instructions = generate_instruction(
            instruction_generator=instruction_predictor,
            image_seqs=batched_image_seqs,
            action_seqs=batched_action_seqs,
            path_mask=path_masks,
            max_instr_len=config.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH,
        )
        dict_episode = {
            'episode_id': episode.episode_id,
            'scene_id': episode.scene_id,
            'start_position': episode.start_position,
            'start_rotation': episode.start_rotation,
            'info': episode.info,
            'goals': [
                {
                    'position': goal.position,
                    'radius': goal.radius,
                    'object_id': goal.object_id,
                    'object_name': goal.object_name,
                    'object_category': goal.object_category,
                    'room_id': goal.room_id,
                    'room_name': goal.room_name,
                    'view_points': [loc.agent_state.position for loc in goal.view_points],
                } for goal in episode.goals
            ],
            'start_room': episode.start_room,
            'shortest_paths': episode.shortest_paths,
            'object_category': episode.object_category,
            'sound_id': episode.sound_id,
            'offset': episode.offset,
            'duration': episode.duration,
            'instructions': instructions.cpu().numpy().tolist(),
        }
        dict_dataset[f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz"]['episodes'].append(dict_episode)
        if cnt > 100000:
            break
        cnt += 1

        # f = open("debug.txt", "a")
        # f.write(f"time: {time.time() - s}\n")
        # f.close()
    
    for key, values in dict_dataset.items():
        print(f"len of {key}: {len(values['episodes'])}\n")
        dataset_json_str = json.dumps(values)
        with open(f"{save_dataset_path}/content/{key}", "wt") as f:
            f.write(dataset_json_str)
    

if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"start make_instructions!\n")
    f.close()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
    )
    parser.add_argument(
        "--save-dataset-path",
        type=str,
    )
    args = parser.parse_args()
    config = get_config(args.config)
    content_scenes_path = "{data_path}/content".format(
        data_path=os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(
            version=config.TASK_CONFIG.DATASET.VERSION,
            split=config.TASK_CONFIG.DATASET.SPLIT,
        )),
    )
    os.makedirs(f"{args.save_dataset_path}/content", exist_ok=True)

    main(config.TASK_CONFIG, config.SENSORS, content_scenes_path, args.save_dataset_path)


# simlator: 0.001476287841796875
# get_obs_seqs: 0.036184072494506836
# batch: 0.145219087600708
# generate: 1.3487656116485596
# epi: 0.00015783309936523438
# time: 1.5374341011047363