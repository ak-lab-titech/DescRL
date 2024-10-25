from typing import List
import sys
import os
import random

import torch
import numpy as np

sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")
sys.path.append("/home/4/ud02274/navigation/myss/habitat-lab")
sys.path.append("/home/4/ud02274/navigation/myss")

from habitat.sims import make_sim
from habitat.datasets import make_dataset
from scripts.make_vlnce_test_dataset import get_obs_seqs, get_obs_seqs_for_xpred
from ss_baselines.savi.config.default import get_config as ss_get_config
from ss_baselines.savi.iprl_pretraining.offpolicy.off_policy_train import setup_instruction_predictor
from ss_baselines.savi.iprl_pretraining.scripts.make_instructions import setup_instruction_generator
from soundspaces.tasks.semantic_audionav_task import merge_sim_episode_config as ss_merge_sim_episode_config
from soundspaces.utils import generate_video
from xgenerator.common.lang import tokens2sentences, VIDEO_LLAMA2_TOKENIZER
from videollama2.mm_utils import get_model_name_from_path
from videollama2.model.builder import load_pretrained_model
from videollama2.train import process_video
from videollama2.constants import MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token


def fix_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_config(environment_type: str, dir_path: str = "data/tmp"):
    if environment_type == "ss1-savi":
        config_path = "ss_baselines/savi/config/semantic_audionav/savi.yaml"
        config = ss_get_config(config_path, None, dir_path, 'train', False)
    else:
        raise Exception(f"environment_type: {environment_type}")
    return config


def setup_model(environment_type: str, model_type: str):
    config = setup_config(environment_type)

    if model_type == "xgen-cnntf":
        assert config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MODEL_TYPE == "cnn_tf"
        model = setup_instruction_generator(
            xgenerator_path=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_PATH,
            ckpt_num=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_CKPT,
        )
        processor = None
    elif model_type == "savi-past-xpred":
        assert environment_type == "ss1-savi", f"Not implemented, {environment_type}"

        if config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MODEL_TYPE == "cnn_tf":
            visual_encoder_output_size=512-4
        elif config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MODEL_TYPE == "video_llama2":
            visual_encoder_output_size=1024
        else:
            raise Exception(f"model_type: {config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MODEL_TYPE}")

        config.defrost()
        config.RL.DDPPO.pretrained = False
        config.freeze()
        
        model, _ = setup_instruction_predictor(
            config=config,
            device="cuda",
            visual_encoder_output_size=visual_encoder_output_size,
            tokenizer_type=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.TOKENIZER_TYPE,
            environment_type=environment_type,
        )

        pretrained_state = torch.load(config.RL.DDPPO.pretrained_weights, map_location="cpu")
        pretrained_state_dict = {}

        for k, v in pretrained_state["belief_predictor"].items():
            pretrained_state_dict[f"belief_predictor.{k}"] = v
        for k, v in pretrained_state["state_dict"].items():
            if "actor_critic.action_distribution" in k or "actor_critic.critic" in k or "task_embedding_for_RL" in k or "ac_encoder" in k:
                continue
            pretrained_state_dict[k[len("actor_critic.net."):]] = v
        
        model.load_state_dict(pretrained_state_dict)
        model.to(torch.device("cuda", 0))
        model.eval()
        processor = None
    elif model_type == "video_llama2":
        model_path = config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.XGENERATOR_PATH
        _, model, processor, _ = load_pretrained_model(
            model_path,
            None,
            get_model_name_from_path(model_path),
        )
    else:
        raise Exception(f"model_type: {model_type}")
    
    return model, processor


def setup_episodes_and_sim(environment_type: str):
    config = setup_config(environment_type)

    if environment_type == "ss1-savi":
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
    else:
        raise Exception(f"environment_type: {environment_type}")
    
    return episodes, sim, sim_cfg


def main(
    model_type: str,
    environment_type: str,
    indices: List[int],
    beam_num: int,
    top_k: int,
    top_p: float,
    temperature: float,
    save_dir_path: str,
):
    os.makedirs(save_dir_path, exist_ok=True)
    with open(f"{save_dir_path}/explanation.txt", "w") as f:
        f.write(f"model_type: {model_type}\n")
        f.write(f"environment_type: {environment_type}\n")
        f.write(f"indices: {indices}\n")
        f.write(f"beam_num: {beam_num}\n")
        f.write(f"top_k: {top_k}\n")
        f.write(f"top_p: {top_p}\n")
        f.write(f"temperature: {temperature}\n")
    
    config = setup_config(environment_type, save_dir_path)
    model, processor = setup_model(environment_type, model_type)
    episodes, sim, sim_cfg = setup_episodes_and_sim(environment_type)

    # Evaluate the model
    for index in indices:
        episode = episodes[index]

        # Reset simulator
        if environment_type == "ss1-savi":
            sim_cfg = ss_merge_sim_episode_config(sim_cfg, episode)
        else:
            raise Exception(f"environment_type: {environment_type}")
        sim.reconfigure(sim_cfg)
        _ = sim.reset()
        
        # Get observations and generate an explanation
        if model_type == "xgen-cnntf":
            image_seqs, action_seqs = get_obs_seqs(sim, -1)
            tokens = model.generate(
                image_seqs=image_seqs, # (l, b, h, w, c)
                action_seqs=action_seqs, # (l, b, 4)
                max_instr_len=config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.MAX_INSTRUCTION_LENGTH,
                save_dir_path=f"{save_dir_path}/episode_{index}_hist",
                beam_num=beam_num,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            image_seqs = image_seqs[:, :, :, :, :3]
        elif model_type == "savi-past-xpred":
            image_seqs, audio_seqs, pose_seqs, action_seqs = get_obs_seqs_for_xpred(sim, episode)
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
            tokens = model.generate(
                observations=inputs,
                prev_actions=inputs["action"],
                save_dir_path=f"{save_dir_path}/episode_{index}_hist",
                # save_dir_path=None,
                beam_num=beam_num,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
            )
            image_seqs = torch.from_numpy(image_seqs[:, :, :, :, :3].copy())
        elif model_type == "video_llama2":
            prompt = "[INST] <<SYS>>\n" \
                    "A chat between a curious user and an artificial intelligence assistant." \
                    "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                    "\n<</SYS>>\n\n <video>\nWhat is the camera wearer doing? [/INST]"
            input_ids = tokenizer_MMODAL_token(
                prompt, VIDEO_LLAMA2_TOKENIZER, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors='pt',
            ).unsqueeze(0).to(device="cuda")
            image_seqs, _ = get_obs_seqs(sim, -1)
            tensor = image_seqs[:, 0, :, :, :3]
            n_frame = len(tensor)
            n_slice = config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.NUM_FRAMES
            indices = np.arange(0, n_frame, n_frame / n_slice).astype(int)
            tensor = tensor[indices]
            tensor = process_video(
                (tensor.to('cpu').detach().numpy().copy() * 255).astype(np.uint8),
                processor,
                "pad",
                n_slice,
            ).to(
                dtype=torch.float16,
                device='cuda',
                non_blocking=True,
            )
            with torch.inference_mode():
                tokens = model.generate(
                    input_ids,
                    images_or_videos=[tensor],
                    modal_list=['video'],
                    do_sample=True,
                    temperature=0.2,
                    # max_new_tokens=1024,
                    max_new_tokens=40,
                    use_cache=True,
                ).view(-1,)
            image_seqs = image_seqs[:, :, :, :, :3]
        else:
            raise Exception(f"model_type: {model_type}")

        # Visualize (visual and auditory) observations
        generate_video(image_seqs, f"{save_dir_path}/image_seqs_{index}.mp4")

        with open(f"{save_dir_path}/explanation.txt", "a") as f:
            f.write(f"-------------- {index} --------------\n")
            f.write(f"Goal object category: {episode.object_category}\n")
            f.write(f"tokens: {tokens}\n")

            tokenizer_type = config.TASK_CONFIG.TASK.GENERATED_INSTRUCTION.TOKENIZER_TYPE
            if tokenizer_type == "r2r":
                sentence = tokens2sentences(tokens.view(-1, 1))[0]
            elif tokenizer_type == "video_llama2":
                sentence = VIDEO_LLAMA2_TOKENIZER.batch_decode(tokens.unsqueeze(0), skip_special_tokens=True)[0]
            else:
                raise Exception(f"tokenizer_type: {tokenizer_type}")
            
            f.write(f"sentence: {sentence}\n")



if __name__=="__main__":
    model_type = "xgen-cnntf" # "xgen-cnntf", "savi-past-xpred", "video_llama2"
    environment_type = "ss1-savi" # "ss1-savi", "habitat-objnav"
    indices = [0, 5]
    beam_num = 1
    top_k = None
    top_p = None
    temperature = 1.0
    save_dir_path = f"data/videos/qual_eval/{environment_type}/{model_type}/beam{beam_num}-k{top_k}-p{top_p}-t{temperature}"

    fix_seeds(seed=0)

    main(
        model_type=model_type,
        environment_type=environment_type,
        indices=indices,
        beam_num=beam_num,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        save_dir_path=save_dir_path,
    )

