import sys
import argparse
import gzip
import json
import os
import time

import yaml
import numpy as np
import torch
from dotmap import DotMap
from tqdm import trange
from transformers import AutoProcessor, VideoLlavaForConditionalGeneration

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")
sys.path.append("/home/4/ud02274/navigation/myss")
sys.path.append("/home/4/ud02274/navigation/my-VLN-CE")
sys.path.append("/home/4/ud02274/navigation/VideoLLaMA2")

from habitat.sims import make_sim
from habitat.datasets import make_dataset
from habitat_sim.utils.common import quat_from_angle_axis
from habitat_baselines.config.default import get_config as habitat_get_config
from ss_baselines.savi.config.default import get_config as ss_get_config
from habitat.tasks.nav.nav import merge_sim_episode_config as habitat_merge_sim_episode_config
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config as ss_merge_sim_episode_config
from xgenerator.common.lang import R2RLang, sentence2token
from common.load_lmdb import PAD_IDX, BOS_IDX, EOS_IDX
from xgenerator.transformer_speaker.model import Seq2SeqTransformer
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import (
    compute_spectrogram,
    compute_pose,
)
from ss_baselines.savi.iprl_pretraining.offpolicy.off_policy_train import setup_instruction_predictor
from soundspaces.utils import generate_video, visualize_spectrogram
from xgenerator.common.lang import R2RLang, tokens2sentences
from xgenerator.common.hugging_face_utils import HFR2RDataset
from xgenerator.video_llava.qlora import video_llava_collate_fn
import vlnce_baselines
from peft import PeftModel
from videollama2.mm_utils import get_model_name_from_path
from videollama2.model.builder import load_pretrained_model
from videollama2.train import process_video
from videollama2.constants import MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token



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


def get_obs_seqs(sim, future_step_num, goal_radius=None, goal_poss=None):
    image_seqs_list = []
    action_seqs_list = []
    if goal_radius is not None and goal_poss is not None:
        oracle_actions = sim.get_oracle_actions_from_current_pos(goal_radius, goal_poss)
    else:
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

    image_seqs = torch.from_numpy(np.array(image_seqs_list)).cuda()
    action_seqs = torch.from_numpy(np.array(action_seqs_list)).cuda()
    
    return image_seqs, action_seqs


def get_obs_seqs_for_xpred(sim, episode):
    image_seqs_list = []
    audio_seqs_list = []
    pose_seqs_list = []
    action_seqs_list = []
    oracle_actions = sim.get_oracle_actions_from_current_pos()
    cnt = 0
    # for文だと-1の時に対応できないのでwhile
    while True:
        
        sim_obs = sim._get_sim_observation()
        observations = sim._sensor_suite.get_observations(sim_obs)
        image_shape = np.shape(observations["depth"])
        
        if len(image_shape) == 4:
            depth_img = np.squeeze(observations["depth"], axis=3)
        else:
            depth_img = observations["depth"]
        rgb_img = observations["rgb"] / 255.0
            
        image = np.concatenate([rgb_img, depth_img], 2).astype(np.float32)
        
        audio = sim.get_current_spectrogram_observation(compute_spectrogram)
        pose = compute_pose(episode, sim.get_agent_state(), cnt)

        if cnt == 0:
            save_action = 0
        else:
            save_action = oracle_actions[cnt-1]
        # action_onehot = np.eye(4)[action].astype(np.int8)

        image_seqs_list.append([image])
        audio_seqs_list.append([audio])
        pose_seqs_list.append([pose])
        action_seqs_list.append([[save_action]])

        if oracle_actions[cnt-1] == 0:
            break
        sim.step(oracle_actions[cnt])
        cnt += 1

    image_seqs = np.array(image_seqs_list)
    audio_seqs = np.array(audio_seqs_list)
    pose_seqs = np.array(pose_seqs_list)
    action_seqs = np.array(action_seqs_list)
    
    return image_seqs, audio_seqs, pose_seqs, action_seqs



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
    past_tokens = past_tokens[past_tokens != 0.0][1:-1].view(-1, 1) # EOS, BOS, PADを外す
    return past_tokens


def generate_instruction_from_xpred(
    instruction_predictor,
    image_seqs,
    audio_seqs,
    pose_seqs,
    action_seqs,
):
    inputs = {
        "rgb": torch.from_numpy(image_seqs[:, :, :, :, :3]).float().cuda(),
        "depth": torch.from_numpy(image_seqs[:, :, :, :, 3:4]).float().cuda(),
        "pose": torch.from_numpy(pose_seqs).cuda(),
        "spectrogram": torch.from_numpy(audio_seqs).float().cuda(),
        "action": torch.from_numpy(action_seqs).cuda(),
        "category": None,
        "location": None,
        "mask": None,
        "seq_lengths": torch.from_numpy(np.array([len(audio_seqs)])).cuda(),
        "target": None,
    }
    logits = instruction_predictor(
        observations=inputs,
        prev_actions=inputs["action"],
        masks=None,
        ext_memory=None,
        ext_memory_masks=inputs["mask"],
    )
    _, tokens = logits.max(2)

    return tokens


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
    instruction_predictor_type, # "xgen", "savi-past-xpred", "savi-future-xpred", "smt-past-xpred", "video-llava"
    environment_type, # "ss1-savi", "habitat-objnav", "vlnce"
    step_num_for_FEPRL=5,
):
    # TODO ここら辺のif文は、メソッドとの中に入れた方が綺麗かもしれない
    if environment_type == "ss1-savi" or environment_type == "habitat-objnav":
        # TODO ここってhabitat-objnavとss1-savi同じでも良いよね？
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
    elif environment_type == "vlnce":
        dataset = make_dataset(
            "VLN-CE-v1",
            config=DotMap({
                "TYPE": "VLN-CE-v1",
                "SPLIT": "val_unseen",
                "DATA_PATH": "../my-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/{split}/{split}.json.gz",
                "SCENES_DIR": "../my-VLN-CE/data/scene_datasets/",
                "CONTENT_SCENES": ['*'],
                "EPISODES_ALLOWED": ['*'],
                "LANGUAGES": ['*'],
                "ROLES": ['guide'],
            }),
        )
        with gzip.open(
            f"../my-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen_gt.json.gz",
            'rb',
        ) as f:
            file_content = f.read()
        data = json.loads(file_content.decode('utf-8'))
        lmdb_id2epi_id = [int(id)-1 for id in list(data.keys())]
        epi_id2lmdb_id = {epi_id: lmdb_id for lmdb_id, epi_id in enumerate(lmdb_id2epi_id)}
        episodes = dataset.episodes
        print(f"length of episodes: {len(episodes)}")
        if instruction_predictor_type == "xgen":
            hf_r2r_dataset = HFR2RDataset(
                data_path="/home/4/ud02274/navigation/my-VLN-CE/data/trajectories_dirs/cma/val_unseen_trajectories.lmdb",
                data_num=len(episodes),
                max_instruction_length=80,
                # n_slice=8,
                need_action_and_depth_semantic=True,
            )
        else:
            num_frames = config.NUM_FRAMES
            hf_r2r_dataset = HFR2RDataset(
                data_path="/home/4/ud02274/navigation/my-VLN-CE/data/trajectories_dirs/cma/val_unseen_trajectories.lmdb",
                data_num=len(episodes),
                max_instruction_length=80,
                n_slice=num_frames,
            )
        # hf_r2r_dataloader = torch.utils.data.DataLoader(
        #     hf_r2r_dataset,
        #     num_workers=1,
        #     batch_size=1,
        #     shuffle=False,
        #     drop_last=False,
        #     collate_fn=video_llava_collate_fn,
        # )
    else:
        raise Exception(f"environment_type: {environment_type}")
    
    if instruction_predictor_type == "xgen":
        instruction_predictor = setup_instruction_generator(
            config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_PATH,
            config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_CKPT,
        )
    elif instruction_predictor_type == "savi-past-xpred" or instruction_predictor_type == "savi-future-xpred":
        config.defrost()
        config.RL.DDPPO.pretrained = False
        config.freeze()
        instruction_predictor, _ = setup_instruction_predictor(
            config=config,
            device=torch.device("cuda", 0),
            visual_encoder_output_size=512-4,
            tokenizer_type="r2r",
            environment_type="ss1-savi",
        )
        pretrained_state = torch.load(config.RL.DDPPO.pretrained_weights, map_location="cpu")
        pretrained_state_dict = {}

        # belief_predictor_state = torch.load("data/models/ss1-savi/mp3d/savi-2nd/past-eprl-v2/data/ckpt.45.pth", map_location="cpu")
        # for k, v in belief_predictor_state["belief_predictor"].items():
        for k, v in pretrained_state["belief_predictor"].items():
            pretrained_state_dict[f"belief_predictor.{k}"] = v
        for k, v in pretrained_state["state_dict"].items():
            if "actor_critic.action_distribution" in k or "actor_critic.critic" in k:
                continue
            # pretrained_state_dict[k] = v
            pretrained_state_dict[k[len("actor_critic.net."):]] = v
        
        instruction_predictor.load_state_dict(pretrained_state_dict)
        instruction_predictor.to(torch.device("cuda", 0))
        instruction_predictor.eval()
    elif instruction_predictor_type == "smt-past-xpred":
        instruction_predictor = None
    elif instruction_predictor_type == "video-llava":
        original_model = config.ORIGINAL_MODEL
        trained_model = config.TRAINED_MODEL

        processor = AutoProcessor.from_pretrained(original_model)
        processor.tokenizer.padding_side = "right"
        instruction_predictor = VideoLlavaForConditionalGeneration.from_pretrained(
            original_model,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        if trained_model is not None:
            instruction_predictor = PeftModel.from_pretrained(instruction_predictor, trained_model)
    elif instruction_predictor_type == "video-llama2":
        model_path = config.MODEL_PATH
        tokenizer, instruction_predictor, processor, _ = load_pretrained_model(
            model_path,
            None,
            get_model_name_from_path(model_path),
        )
    elif instruction_predictor_type == "random":
        lang = R2RLang()
        vocab_size = lang.vocab_size - 2 # bosとeosを除く
    else:
        raise Exception(f"instruction_predictor_type: {instruction_predictor_type}")
    
    with gzip.open(
        "/home/4/ud02274/navigation/my-VLN-CE/data/datasets/R2R_VLNCE_v1-3_preprocessed/val_unseen/val_unseen.json.gz",
        'rb',
    ) as f:
        file_content = f.read()
    val_unseen_data = json.loads(file_content.decode('utf-8'))
    dict_dataset = {"instruction_vocab": val_unseen_data["instruction_vocab"], "episodes": []}

    print(f"length of episodes: {len(episodes)}\n")
    for i in trange(len(episodes)):
        episode = episodes[i]
        print(f"\ri={i+1}/{len(episodes)}", end="")

        if environment_type == "vlnce":
            if instruction_predictor_type == "video-llava":
                video, _ = hf_r2r_dataset[epi_id2lmdb_id[i]]

                inputs = processor(
                    videos=video,
                    text="USER: <video>What is the camera wearer doing? ASSISTANT: ",
                    return_tensors="pt",
                )
                for k, v in inputs.items():
                    inputs[k] = v.to("cuda")
                with torch.inference_mode():
                    instructions = instruction_predictor.generate(max_new_tokens=40, **inputs)
                instructions = processor.batch_decode(instructions, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                instructions = torch.tensor([sentence2token(instructions[0][50:])], dtype=torch.int32).view(-1, 1) # (instr_len, 1)
            elif instruction_predictor_type == "video-llama2":
                video, _ = hf_r2r_dataset[epi_id2lmdb_id[i]]
                prompt = "[INST] <<SYS>>\n" \
                        "A chat between a curious user and an artificial intelligence assistant." \
                        "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                        "\n<</SYS>>\n\n <video>\nWhat is the camera wearer doing? [/INST]"
                input_ids = tokenizer_MMODAL_token(
                    prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors='pt',
                ).unsqueeze(0).to(device="cuda")
                tensor = process_video(
                    video.numpy().copy(),
                    processor,
                    "pad",
                    num_frames,
                ).to(
                    dtype=torch.float16,
                    device='cuda',
                    non_blocking=True,
                )
                with torch.inference_mode():
                    output_ids = instruction_predictor.generate(
                        input_ids,
                        images_or_videos=[tensor],
                        modal_list=['video'],
                        do_sample=True,
                        temperature=0.2,
                        # max_new_tokens=1024,
                        max_new_tokens=40,
                        use_cache=True,
                    )
                instructions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
                instructions = torch.tensor([sentence2token(instructions)], dtype=torch.int32).view(-1, 1) # (instr_len, 1)
            elif instruction_predictor_type == "random":
                instructions = torch.from_numpy(
                    np.random.randint(
                        0, vocab_size,
                        size=(config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH, 1),
                    )
                )
            else:
                image_seqs, action_seqs, target = hf_r2r_dataset[epi_id2lmdb_id[i]]
                image_seqs = image_seqs.unsqueeze(1) # (l, 1, h, w, c), max:1, min:0
                action_seqs = action_seqs.unsqueeze(1) # (l, 1, a)
                
                instructions = generate_instruction(
                    instruction_generator=instruction_predictor,
                    image_seqs=image_seqs.cuda(),
                    action_seqs=action_seqs.cuda(),
                    path_mask=None,
                    max_instr_len=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH,
                ) # (instr_len, 1)
        else:
            if environment_type == "ss1-savi":
                sim_cfg = ss_merge_sim_episode_config(sim_cfg, episode)
            elif environment_type == "habitat-objnav":
                sim_cfg = habitat_merge_sim_episode_config(sim_cfg, episode)

            sim.reconfigure(sim_cfg)
            _ = sim.reset()
            if instruction_predictor_type == "xgen":
                if environment_type == "ss1-savi":
                    image_seqs, action_seqs = get_obs_seqs(sim, -1)
                elif environment_type == "habitat-objnav":

                    goal_radius = 1.0 # TODO
                    goal_poss = [goal.position for goal in episode.goals]
                    print(f"goal_poss", goal_poss)
                    image_seqs, action_seqs = get_obs_seqs(sim, -1, goal_radius, goal_poss)
                instructions = generate_instruction(
                    instruction_generator=instruction_predictor,
                    image_seqs=image_seqs,
                    action_seqs=action_seqs,
                    path_mask=None,
                    max_instr_len=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH,
                )
            elif instruction_predictor_type == "savi-past-xpred" or instruction_predictor_type == "savi-future-xpred":
                image_seqs, audio_seqs, pose_seqs, action_seqs = get_obs_seqs_for_xpred(sim, episode)
                if instruction_predictor_type == "savi-future-xpred":
                    image_seqs = image_seqs[:step_num_for_FEPRL]
                    audio_seqs = audio_seqs[:step_num_for_FEPRL]
                    pose_seqs = pose_seqs[:step_num_for_FEPRL]
                    action_seqs = action_seqs[:step_num_for_FEPRL]
                instructions = generate_instruction_from_xpred(
                    instruction_predictor,
                    image_seqs,
                    audio_seqs,
                    pose_seqs,
                    action_seqs,
                )
            elif instruction_predictor_type == "random":
                instructions = torch.from_numpy(
                    np.random.randint(
                        0, vocab_size,
                        size=(config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH, 1),
                    )
                )

            elif instruction_predictor_type == "smt-past-xpred":
                raise NotImplementedError()
            else:
                raise Exception(f"instruction_predictor_type: {instruction_predictor_type}")
        
        dict_episode = {
            'episode_id': episode.episode_id,
            'trajectory_id': episode.trajectory_id if environment_type == "vlnce" else episode.episode_id, # TODO Is this OK?
            'scene_id': os.path.join(*episode.scene_id.split("/")[-3:]),
            'start_position': episode.start_position,
            'start_rotation': episode.start_rotation,
            'goals': [
                {
                    'position': goal.position,
                    'radius': goal.radius,
                } for goal in episode.goals
            ],
            'instruction': {
                'instruction_text': tokens2sentences(instructions.view(-1, 1))[0],
                'instruction_tokens': instructions.view(-1,).cpu().numpy().tolist(),
            },
            'reference_path': None,
        }
        dict_dataset["episodes"].append(dict_episode)

    print()    
    print(f"len of episodes: {len(dict_dataset['episodes'])}\n")
    dataset_json_str = json.dumps(dict_dataset)
    with gzip.open(f"{save_dataset_path}/{save_dataset_path.split('/')[-1]}.json.gz", "wt") as f:
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
        "--instruction-predictor-type",
        type=str,
    )
    parser.add_argument(
        "--environment-type",
        type=str,
    )
    args = parser.parse_args()

    if args.environment_type == "ss1-savi":
        config = ss_get_config(args.config)
        content_scenes_path = "{data_path}/content".format(
            data_path=os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(
                version=config.TASK_CONFIG.DATASET.VERSION,
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )),
        )
    elif args.environment_type == "habitat-objnav":
        config = habitat_get_config(args.config, model_dir="./habitat-lab/data/models/make_vlnce_test_dataset/hoge")
        content_scenes_path = "{data_path}/content".format(
            data_path=os.path.dirname(config.TASK_CONFIG.DATASET.DATA_PATH.format(
                split=config.TASK_CONFIG.DATASET.SPLIT,
            )),
        )
    else:
        # config = None
        config = ss_get_config(args.config)
        content_scenes_path = None
    
    
    os.makedirs(f"{args.save_dataset_path}/content", exist_ok=True)

    main(
        config,
        content_scenes_path,
        args.save_dataset_path,
        args.instruction_predictor_type,
        args.environment_type,
    )

