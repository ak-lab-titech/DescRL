import sys
import argparse
import random

import yaml
import torch

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from train import rollout
from model import Seq2SeqTransformer
from common.load_lmdb import R2RDataset, my_collate_fn
from common.lang import R2RLang, tokens2sentences



def load_model(model_path, ckpt_num, train_config, lang):
    seq2seq_model = Seq2SeqTransformer(
        num_encoder_layers=train_config["model"]["num_encoder_layers"],   
        num_decoder_layers=train_config["model"]["num_decoder_layers"],
        emb_size=train_config["model"]["emb_size"],
        vocab_emb_size=train_config["model"]["vocab_embedding_size"],
        nhead=train_config["model"]["nhead"],
        use_image_feature=train_config["train"]["use_image_feature"],
        vocab_size=lang.vocab_size,
        glove=lang.glove_vec,
        dim_feedforward=train_config["model"]["dim_feedforward"],
        dropout=train_config["model"]["dropout_ratio"],
    )
    seq2seq_model.load_state_dict(torch.load(f"{model_path}/data/{ckpt_num}/seq2seq.pth", torch.device("cpu")))
    return seq2seq_model


def eval_by_a_dataset(
    eval_num,
    seq2seq_model,
    lang,
    data_path,
    use_image_feature,
    data_num,
    skip_frame_per,
    max_instruction_length,
):
    dataset = R2RDataset(
        data_path,
        use_image_feature,
        data_num,
        True,
        skip_frame_per,
        max_instruction_length,
    )
    idxs = [random.randint(0, data_num - 1) for _ in range(eval_num)]
    batch = [dataset[idx] for idx in idxs]
    inputs, targets = my_collate_fn(batch)

    loss, words, target_words = rollout(
        seq2seq_model=seq2seq_model,
        inputs=inputs,
        targets=targets,
        max_instruction_length=max_instruction_length,
        feedback="student",
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
    seq2seq_model = load_model(model_path, ckpt_num, train_config, lang)
    seq2seq_model.eval()

    print("Seen")
    eval_by_a_dataset(
        eval_num,
        seq2seq_model,
        lang,
        train_config["train"]["val_seen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_seen_data_num"],
        train_config["train"]["skip_frame_per"],
        train_config["train"]["max_instruction_length"],
    )
    print()
    print("Unseen")
    eval_by_a_dataset(
        eval_num,
        seq2seq_model,
        lang,
        train_config["train"]["val_unseen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_unseen_data_num"],
        train_config["train"]["skip_frame_per"],
        train_config["train"]["max_instruction_length"],
    )



if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START eval transformer-speaker\n")
    f.close()
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
