import sys
import time

import tensorflow as tf
import tensorflow_hub as hub
import torch
import numpy as np
import seaborn as sns
from gym import spaces


sys.path.insert(0, "/home/0/19B30511/av-nav/myss/sound-spaces")
sys.path.append("/home/0/19B30511/av-nav/myss/habitat-lab")
sys.path.append("/home/0/19B30511/av-nav/myss")

from habitat.datasets import make_dataset
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_lmdb_dataset import IPRLPretrainingLMDBDataset
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_dataset import compute_spectrogram
from ss_baselines.savi.iprl_pretraining.offpolicy.off_policy_train import setup_instruction_predictor
from ss_baselines.savi.iprl_pretraining.offpolicy.off_policy_eval import get_data
from ss_baselines.savi.config.default import get_config
from xgenerator.common.load_lmdb import PAD_IDX
from xgenerator.common.lang import tokens2sentences, R2RLang

np.random.seed(0)


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_instruction_predictor(config):
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
        device=torch.device("cuda", 0),
        observation_spaces=observation_space,
        action_spaces=action_space,
        has_distractor_sound=config.TASK_CONFIG.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND,
        pretrained=config.RL.DDPPO.pretrained,
        pretrained_weights=config.RL.DDPPO.pretrained_weights,
    )
    return instruction_predictor, loss_fn

def arrange_sentence(sentence):
    sentence = sentence.split(" ")
    if sentence[0] == "<bos>":
        sentence = sentence[1:]
    
    for i in range(len(sentence)):
        if sentence[i] == "<eos>":
            sentence = sentence[:i]
            break
    
    arranged_sentence = ""
    for i in range(len(sentence)):
        if i == 0:
            arranged_sentence = sentence[i]
        else:
            if sentence[i] == "." or sentence[i] == ",":
                arranged_sentence = arranged_sentence + sentence[i]
            else:
                arranged_sentence = arranged_sentence + " " + sentence[i]
    
    return arranged_sentence
    

def main():
    n_iters = 1000

    config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml")

    lang = R2RLang("r2r")
    print(f"dataset config")
    print(config.TASK_CONFIG.DATASET)

    dataset = make_dataset(
        id_dataset=config.TASK_CONFIG.DATASET.TYPE,
        config=config.TASK_CONFIG.DATASET,
    )
    episodes = dataset.episodes

    instruction_predictor, loss_fn = get_instruction_predictor(config)

    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    use = hub.load(module_url)

    # sentence_1 = ["I am a cat."]
    # sentence_2 = ["I am a cat ."]
    # sentence_3 = ["I am a dog."]
    # sentence_4 = ["I am the cat."]
    # sentence_5 = ["She is a cat."]
    # sentence_6 = ["She is a dog."]
    # sentence_7 = ["I like to play baseball."]
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_2)[0])}")
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_3)[0])}")
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_4)[0])}")
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_5)[0])}")
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_6)[0])}")
    # print(f"cos sim: {cos_sim(use(sentence_1)[0], use(sentence_7)[0])}")
    # cos sim: 1.0
    # cos sim: 0.6529510617256165
    # cos sim: 0.8809022307395935
    # cos sim: 0.7032529711723328
    # cos sim: 0.4032006859779358
    # cos sim: 0.32287344336509705

    sims = []
    losses = []
    s = time.time()
    for i in range(n_iters):
        print(f"\r{i+1}/{n_iters}", end="")
        data = get_data(
            index=i,
            episode=episodes[i],
            dataset_path=f"./data/lmdb_dataset/iprl_pretrain_test",
            gen_video=False,
            model_dir=None,
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
        losses.append(loss.item())

        _, iprl_tokens = logits.max(2)
        pred_sentence = tokens2sentences(iprl_tokens, lang)[0]
        true_sentence = tokens2sentences(data["target"].permute(1, 0), lang)[0]

        pred_sentence = arrange_sentence(pred_sentence)
        true_sentence = arrange_sentence(true_sentence)
        
        
        sim = cos_sim(use([pred_sentence])[0], use([true_sentence])[0])
        sims.append(sim)

        # print(f"pred sentence: {pred_sentence}")
        # print(f"true sentence: {true_sentence}")
        # print(f"similarity: {sim}")
    print()
    print(f"mean similarity: {np.mean(sims)}")
    print(f"mean cross entropy loss: {np.mean(losses)}")
    print(f"TIME: {(time.time() - s) / 60} [min]")


if __name__=="__main__":
    main()
