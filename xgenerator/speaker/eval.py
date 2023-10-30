import sys
import argparse
import random

import yaml
import torch

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from train import rollout
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM
from common.load_lmdb import R2RDataset, my_collate_fn
from common.lang import R2RLang



def load_model(model_path, ckpt_num, train_config, lang):
    encoder = SpeakerEncoderLSTM(
        action_embedding_size=4,
        world_embedding_size=train_config["model"]["world_embedding_size"],
        hidden_size=train_config["model"]["hidden_size"],
        dropout_ratio=train_config["model"]["dropout_ratio"],
        use_image_feature=train_config["train"]["use_image_feature"],
        bidirectional=train_config["model"]["bidirectional"],
    )
    decoder = SpeakerDecoderLSTM(
        vocab_size=lang.vocab_size,
        vocab_embedding_size=train_config["model"]["vocab_embedding_size"],
        hidden_size=train_config["model"]["hidden_size"],
        dropout_ratio=train_config["model"]["dropout_ratio"],
        glove=lang.glove_vec,
        use_input_att_feed=train_config["model"]["use_input_att_feed"],
    )
    encoder.load_state_dict(torch.load(f"{model_path}/{ckpt_num}/encoder.pth", torch.device("cpu")))
    decoder.load_state_dict(torch.load(f"{model_path}/{ckpt_num}/decoder.pth", torch.device("cpu")))
    return encoder, decoder


def tokens2sentences(tokens, lang):
    sentences = ["" for _ in range(len(tokens[0]))]
    for token in tokens:
        token = token.to('cpu').detach().numpy().copy()
        for i in range(len(token)):
            word = lang.index2word[token[i]]
            sentences[i] = sentences[i] + word + " "
    return sentences


def eval_by_a_dataset(
    eval_num,
    encoder,
    decoder,
    lang,
    data_path,
    use_image_feature,
    data_num,
    max_instruction_length,
):
    dataset = R2RDataset(
        data_path,
        use_image_feature,
        data_num,
    )
    idxs = [random.randint(0, data_num - 1) for _ in range(eval_num)]
    batch = [dataset[idx] for idx in idxs]
    inputs, targets = my_collate_fn(batch)

    loss, words, target_words = rollout(
        encoder=encoder,
        decoder=decoder,
        inputs=inputs,
        targets=targets,
        max_instruction_length=max_instruction_length,
        feedback="argmax",
        return_words=True,
    )
    pred_sentences = tokens2sentences(words, lang)
    true_sentences = tokens2sentences(target_words, lang)
    print(f"Loss: {loss.item()}")
    for i in range(eval_num):
        print(f"------- {i+1}/{eval_num} -------")
        print(f"index: {idxs[i]}")
        print(f"Predict:\n{pred_sentences[i]}")
        print(f"True:\n{true_sentences[i]}")

def eval(
    eval_num,
    model_path,
    ckpt_num,
    train_config,
):
    lang = R2RLang(name="r2r_train")
    encoder, decoder = load_model(model_path, ckpt_num, train_config, lang)
    encoder.eval()
    decoder.eval()

    print("Seen")
    eval_by_a_dataset(
        eval_num,
        encoder,
        decoder,
        lang,
        train_config["train"]["val_seen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_seen_data_num"],
        train_config["train"]["max_instruction_length"],
    )
    print()
    print("Unseen")
    eval_by_a_dataset(
        eval_num,
        encoder,
        decoder,
        lang,
        train_config["train"]["val_unseen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_unseen_data_num"],
        train_config["train"]["max_instruction_length"],
    )



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-num', help='the number of evaluation.')
    parser.add_argument('--ckpt-num', help='the number of checkpoint.')
    parser.add_argument('--model-path', help='the path of trained model.')
    parser.add_argument('--random-seed', help="random seed.")
    
    args = parser.parse_args()
    
    with open(f"{args.model_path}/config.yaml", "r") as yml:
        train_config = yaml.safe_load(yml)
    
    random.seed(int(args.random_seed))
    
    eval(
        eval_num=int(args.eval_num),
        model_path=args.model_path,
        ckpt_num=int(args.ckpt_num),
        train_config=train_config,
    )
