import os
import sys
import argparse
import random
import logging

import yaml
import torch
import numpy as np
from moviepy.editor import ImageSequenceClip

sys.path.append("/home/4/ud02274/navigation/myss/xgenerator")

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
        use_semantic=train_config["model"]["use_semantic"],
        glove=lang.glove_vec,
        dim_feedforward=train_config["model"]["dim_feedforward"],
        dropout=train_config["model"]["dropout_ratio"],
        use_lfao=config["model"]["lfao"]["use_lfao"],
        num_lfao_decoder_layers=config["model"]["lfao"]["lfao_num_decoder_layers"],
    )
    seq2seq_model.load_state_dict(torch.load(f"{model_path}/data/{ckpt_num}/seq2seq.pth", torch.device("cpu")))
    seq2seq_model = seq2seq_model.to("cuda")
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
    logger,
    save_dir_path,
    beam_num,
    top_k,
    top_p,
    temperature,
    video_dir_path=None,
):
    dataset = R2RDataset(
        data_path,
        use_image_feature,
        data_num,
        True,
        skip_frame_per,
        max_instruction_length,
        True,
        False,
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
    
    image_seqs = torch.from_numpy(np.array(inputs["image_seqs"])).cuda()
    action_seqs = torch.from_numpy(np.array(inputs["action_seqs"])).cuda()
    path_mask = torch.from_numpy(inputs["mask"]).cuda()

    pred_sentences = tokens2sentences(words, lang)
    true_sentences = tokens2sentences(target_words, lang)
    logger.info(f"Loss: {loss.item()}")
    for i in range(eval_num):
        logger.info(f"------- {i+1}/{eval_num} -------")
        logger.info(f"Index: {idxs[i]}")
        logger.info(f"True: {true_sentences[i]}")
        logger.info(f"Predict: {pred_sentences[i]}")

        if video_dir_path is not None:
            image_seq = image_seqs[:, i, :, :, :]*255
            image_seq = [image_seq[j].cpu().detach().numpy().copy().astype(np.uint8)[:, :, :3] for j in range(len(image_seq)) if j < inputs["seq_lengths"][i]]
            clip = ImageSequenceClip(image_seq, fps=2)
            clip.write_videofile(f"{video_dir_path}/{i}.mp4", codec="libx264")

        generated_words = seq2seq_model.generate(
            image_seqs=image_seqs[:, i:i+1, :, :, :], # (l, b, h, w, c)
            action_seqs=action_seqs[:, i:i+1, :], # (l, b, 4)
            max_instr_len=max_instruction_length,
            # save_dir_path=save_dir_path,
            save_dir_path=None,
            beam_num=beam_num,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            src_padding_mask=path_mask[i:i+1, :], # (b, l)
        ).view(-1, 1)
        logger.info(f"Generate (i): {tokens2sentences(generated_words, lang)}")


def eval(
    eval_num,
    model_path,
    ckpt_num,
    train_config,
    logger,
    beam_num,
    top_k,
    top_p,
    temperature,
):
    lang = R2RLang(name="r2r_train")
    seq2seq_model = load_model(model_path, ckpt_num, train_config, lang)
    seq2seq_model.eval()

    os.makedirs(f"{model_path}/video/unseen", exist_ok=True)
    os.makedirs(f"{model_path}/video/seen", exist_ok=True)
    logger.info("Seen")
    eval_by_a_dataset(
        eval_num,
        seq2seq_model,
        lang,
        train_config["train"]["val_seen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_seen_data_num"],
        train_config["train"]["skip_frame_per"],
        train_config["train"]["max_instruction_length"],
        logger,
        model_path,
        beam_num,
        top_k,
        top_p,
        temperature,
        video_dir_path=f"{model_path}/video/seen",
    )
    logger.info("Unseen")
    eval_by_a_dataset(
        eval_num,
        seq2seq_model,
        lang,
        train_config["train"]["val_unseen_data_path"],
        train_config["train"]["use_image_feature"],
        train_config["train"]["val_unseen_data_num"],
        train_config["train"]["skip_frame_per"],
        train_config["train"]["max_instruction_length"],
        logger,
        model_path,
        beam_num,
        top_k,
        top_p,
        temperature,
        video_dir_path=f"{model_path}/video/unseen",
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
    parser.add_argument('--beam-num', default=1, type=int)
    parser.add_argument('--top-k', default=1, type=int)
    parser.add_argument('--top-p', nargs='?', default=None, type=float)
    parser.add_argument('--temperature', default=1, type=float)
    
    args = parser.parse_args()
    
    with open(f"{args.model_path}/config.yaml", "r") as yml:
        train_config = yaml.safe_load(yml)
    
    random.seed(int(args.random_seed))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{args.model_path}/eval_{args.ckpt_num}.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Start!")

    logger.info(f"beam_num: {args.beam_num}")
    logger.info(f"top_k: {args.top_k}")
    logger.info(f"top_p: {args.top_p}")
    logger.info(f"temperature: {args.temperature}")
    
    eval(
        eval_num=int(args.eval_num),
        model_path=args.model_path,
        ckpt_num=int(args.ckpt_num),
        train_config=train_config,
        logger=logger,
        beam_num=args.beam_num,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
    )
