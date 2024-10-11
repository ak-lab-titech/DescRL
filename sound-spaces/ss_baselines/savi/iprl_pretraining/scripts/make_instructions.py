import sys
import argparse
import gzip
import json
import os
import time

import yaml
import numpy as np
import torch
from tqdm import trange

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")
sys.path.append("/home/4/ud02274/navigation/myss")

from habitat.sims import make_sim
from habitat.datasets import make_dataset
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_baselines.config.default import get_config as habitat_get_config
from ss_baselines.savi.config.default import get_config as ss_get_config
from habitat.tasks.nav.nav import merge_sim_episode_config as habitat_merge_sim_episode_config
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config as ss_merge_sim_episode_config
from xgenerator.common.lang import R2RLang
from common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX
from xgenerator.transformer_speaker.model import Seq2SeqTransformer
from soundspaces.utils import generate_video, visualize_spectrogram
from xgenerator.common.lang import R2RLang, tokens2sentences


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

    with open(f"{xgenerator_path}/config.yaml", "r") as yml:
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
        use_lfao=config["model"]["lfao"]["use_lfao"],
        num_lfao_decoder_layers=config["model"]["lfao"]["lfao_num_decoder_layers"],
    )
    instruction_generator.load_state_dict(
        torch.load(
            f"{xgenerator_path}/data/{ckpt_num}/seq2seq.pth",
            map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'),
            )
    )
    if torch.cuda.is_available():
        instruction_generator = instruction_generator.to(torch.device("cuda"))
    instruction_generator.eval()
    return instruction_generator


def get_obs_seqs(sim, future_step_num):
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

    image_seqs = np.array(image_seqs_list)
    action_seqs = np.array(action_seqs_list)
    
    return image_seqs, action_seqs


def get_obs_seqs_for_habitat_objnav(sim, oracle_actions):
    # oracle_actions = sim.get_oracle_actions_from_current_pos(
    #     goal_radius=1.0,
    #     view_points=view_points,
    # )
    for i, oracle_action in enumerate(oracle_actions[:-1]):
        sim.step(oracle_action)

    image_seqs = np.array(sim.k_prev_images)
    if len(sim.k_prev_actions) == 0:
        action_seqs = np.eye(4)[0].astype(np.int8).reshape(1, 1, 4)
    elif len(sim.k_prev_actions) < sim.prev_k:
        action_seqs = np.array(list(sim.k_prev_actions) + [[np.eye(4)[0].astype(np.int8)]])
    else:
        action_seqs = np.array(list(sim.k_prev_actions)[1:] + [[np.eye(4)[0].astype(np.int8)]])
    
    return image_seqs, action_seqs


def make_batch(image_seqs, action_seqs, input_future_step_num, future_or_past):
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
        if future_or_past == "future":
            image_seq = image_seqs[i:i+input_future_step_num, 0]
            action_seq = action_seqs[i:i+input_future_step_num, 0]
        elif future_or_past == "past":
            image_seq = image_seqs[max(0, i-input_future_step_num+1):i+1, 0]
            action_seq = action_seqs[max(0, i-input_future_step_num+1):i+1, 0]
        else:
            raise Exception(f"future_or_past: {future_or_past}")
        seq_len = np.shape(image_seq)[0]
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
        
        for i in range(max_instr_len-1):
            tgt_mask = torch.triu(torch.full((i+1, i+1), 1), diagonal=1).type(torch.bool)
            if torch.cuda.is_available():
                tgt_mask = tgt_mask.cuda()
            logits = instruction_generator.decode(
                trg=past_tokens,
                memory=memory,
                tgt_mask=tgt_mask,
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


def make_video(image_seq, audio_seq, pose_seq, action_seq, instruction, output_dir):
    os.makedirs(output_dir, exist_ok=True)  
    os.makedirs(f"{output_dir}/spectrograms", exist_ok=True)
        
    words = tokens2sentences(instruction, R2RLang("r2r_lang"))
        
    f = open(f"{output_dir}/output.txt", "w")
    f.write(f"0: found, 1: forward, 2: left, 3: right\n")
    f.write(f"action_seq: {action_seq}\n")
    f.write(f"pose_seq: {pose_seq}\n")
    f.write(f"instructions: {words}\n")
    f.close()

    generate_video(torch.from_numpy(image_seq.copy()), f"{output_dir}/image_seq.mp4")

    for i in range(len(audio_seq)):
        visualize_spectrogram(audio_seq[i][0], f"{output_dir}/spectrograms/spectrogram_{i}.png")


def main(
    config,
    content_scenes_path,
    save_dataset_path,
    future_or_past,
    environment_type,
):
    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    episodes = dataset.episodes
    sim_cfg = config.TASK_CONFIG.SIMULATOR
    sim_cfg.defrost()
    sim_cfg.SCENE_DATASET = episodes[0].scene_dataset_config
    sim_cfg.AGENT_0.SENSORS = config.SENSORS
    sim_cfg.SCENE = episodes[0].scene_id
    sim_cfg.freeze()
    sim = make_sim(
        sim_cfg.TYPE, config=sim_cfg,
    )
    instruction_predictor = setup_instruction_generator(
        config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_PATH,
        config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_CKPT,
    )

    scene_file_names = [
        f for f in os.listdir(content_scenes_path) if os.path.isfile(os.path.join(content_scenes_path, f))
    ]
    print(f"scene_file_names: {scene_file_names}")
    # for train
    dict_dataset = {f: {'episodes': [], 'scene': f.split('/')[0]} for f in scene_file_names}

    # for val & test
    # dict_dataset = {}

    # step_num_for_FEPRL = 5 # for F-EPRL
    print(f"length of episodes: {len(episodes)}\n")

    if environment_type == "ss1-savi":
        indices = np.arange(len(episodes))
    elif environment_type == "habitat-objnav":
        n_episode = 1000 # for train
        # n_episode = len(episodes) # for val & test
        indices = np.random.choice(np.arange(len(episodes)), size=n_episode, replace=False)
    else:
        raise Exception(f"environment_type: {environment_type}")
    
    for j in trange(len(indices)):
        i = indices[j]
        episode = episodes[i]
        if environment_type == "ss1-savi":
            sim_cfg = ss_merge_sim_episode_config(sim_cfg, episode)
        elif environment_type == "habitat-objnav":
            sim_cfg = habitat_merge_sim_episode_config(sim_cfg, episode)
            oracle_actions = []
            for point in episode.shortest_paths[0]:
                oracle_actions.append(point.action)
            oracle_actions = oracle_actions[:-1] + [0] # 必ず最後がNoneになっているので0にする
            sim.prev_k = len(oracle_actions)
        
        sim.reconfigure(sim_cfg)
        _ = sim.reset()
        if environment_type == "habitat-objnav":
            # view_points = [
            #     view_point.agent_state.position
            #     for goal in episode.goals
            #     for view_point in goal.view_points
            # ]
            # view_points = [episode.info['best_viewpoint_position']]
            image_seqs, action_seqs = get_obs_seqs_for_habitat_objnav(sim, oracle_actions)
        else:
            image_seqs, action_seqs = get_obs_seqs(sim, -1)

        # for XGenerator with batched
        batched_image_seqs, batched_action_seqs, path_masks = make_batch(
            image_seqs,
            action_seqs,
            config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.FUTURE_STEP_NUM,
            future_or_past,
        )

        # for XGenerator
        instructions = generate_instruction(
            instruction_generator=instruction_predictor,
            image_seqs=batched_image_seqs,
            action_seqs=batched_action_seqs,
            path_mask=None,
            # path_mask=path_masks,
            max_instr_len=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH,
        )
        
        episode.info["num_action"] = np.shape(batched_image_seqs)[1] # 元々保存されているnum_actionとoracle_action lengthが異なっている時がある
        if environment_type == "ss1-savi":
            dict_episode = {
                'episode_id': episode.episode_id,
                'scene_id': os.path.join(*episode.scene_id.split("/")[-2:]),
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
                "sound_id": episode.sound_id,
                "offset": episode.offset,
                "duration": episode.duration,
                'object_category': episode.object_category,
                'instructions': instructions.cpu().numpy().tolist(),
            }
        elif environment_type == "habitat-objnav":
            sps = []
            for sp in episode.shortest_paths:
                list_sp = []
                for point in sp:
                    list_sp.append(point.action)
                sps.append(list_sp)

            dict_episode = {
                'episode_id': episode.episode_id,
                'scene_id': os.path.join(*episode.scene_id.split("/")[-3:]),
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
                        'view_points': goal.view_points,
                    } for goal in episode.goals
                ],
                'start_room': episode.start_room,
                'shortest_paths': sps,
                'object_category': episode.object_category,
                'instructions': instructions.cpu().numpy().tolist(),
            }
        
        # for train
        dict_dataset[f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz"]['episodes'].append(dict_episode)

        # for test & val
        # if f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz" in dict_dataset.keys():
        #     dict_dataset[f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz"]['episodes'].append(dict_episode)
        # else:
        #     dict_dataset[f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz"] = {'episodes': [dict_episode], 'scene': f"{sim_cfg.SCENE.split('/')[-1].split('.')[0]}.json.gz"}
    
    for key, values in dict_dataset.items():
        print(f"len of {key}: {len(values['episodes'])}\n")
        dataset_json_str = json.dumps(values)
        with gzip.open(f"{save_dataset_path}/content/{key}", "wt") as f:
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
    parser.add_argument(
        "--future-or-past",
        type=str,
    )
    args = parser.parse_args()
    if "ss_baselines" in args.config:
        config = ss_get_config(args.config)
        content_scenes_path = "{data_path}/content".format(
            data_path=os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(
                version=config.TASK_CONFIG.DATASET.VERSION,
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )),
        )
        environment_type = "ss1-savi"
    elif "habitat_baselines" in args.config:
        config = habitat_get_config(args.config, None, "make_instructions")
        content_scenes_path = "{data_path}/content".format(
            data_path=os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )),
        )
        environment_type = "habitat-objnav"
    else:
        raise Exception(f"args.config: {args.config}")
    
    os.makedirs(f"{args.save_dataset_path}/content", exist_ok=True)

    main(config, content_scenes_path, args.save_dataset_path, args.future_or_past, environment_type)


# simlator: 0.001476287841796875
# get_obs_seqs: 0.036184072494506836
# batch: 0.145219087600708
# generate: 1.3487656116485596
# epi: 0.00015783309936523438
# time: 1.5374341011047363