import os
import sys
import time
import argparse
import logging
import contextlib

import yaml
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D
import torch.multiprocessing as multiprocessing
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import all_reduce

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.utils import try_cuda
from common.lang import R2RLang
from common.load_lmdb import R2RDataset, my_collate_fn, PAD_IDX, BOS_IDX, EOS_IDX
from model import Seq2SeqTransformer


def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]


def save_model(seq2seq_model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_path = f"{save_dir}/seq2seq.pth"
    torch.save(seq2seq_model.module.state_dict(), model_path)


def rollout(seq2seq_model, inputs, targets, max_instruction_length=80, feedback="teacher", return_words=False):
    action_embeddings = try_cuda(Variable(
            torch.from_numpy(np.array(inputs["action_seqs"])),
            requires_grad=False,
    ))
    image_features = try_cuda(Variable(
            torch.from_numpy(np.array(inputs["image_seqs"])),
            requires_grad=False,
    ))
    path_mask = try_cuda(torch.from_numpy(inputs["mask"]))
    targets = try_cuda(torch.from_numpy(np.array(targets))).permute(1,0)
    
    image_seqs_max_l= len(image_features)
    word_seqs_max_l = len(targets)-1
    
    if feedback == "teacher":
        logits = seq2seq_model(
            src_image=image_features, # (max_l, b, image_shape)
            src_action=action_embeddings,      # (max_l, b, 4)
            trg=targets[:-1, :], # (199, b)
            src_mask=torch.zeros((image_seqs_max_l, image_seqs_max_l), dtype=bool), # (max_l, max_l)
            memory_mask=None, # (max_l, 200)?
            tgt_mask=torch.triu(torch.full((word_seqs_max_l, word_seqs_max_l), 1), diagonal=1).type(torch.bool),
            src_padding_mask=path_mask, # (b, max_l)
            tgt_padding_mask=(targets[:-1 :].permute(1,0)==PAD_IDX), # (b, 199)
            memory_key_padding_mask=path_mask, # (b, max_l)
        )
    elif feedback == "student":
        memory = seq2seq_model.encode(
            src_image=image_features,
            src_action=action_embeddings,
            src_mask=torch.zeros((image_seqs_max_l, image_seqs_max_l), dtype=bool),
            src_padding_mask=path_mask,
        ) # (max_l, b, hidden)
        _, batch_size = targets.shape
        preds = torch.full((1, batch_size), BOS_IDX)
        logits_list = []
        for _ in range(max_instruction_length):
            logits = seq2seq_model.decode(
                trg=preds,
                memory=memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_padding_mask=None,
                memory_key_padding_mask=path_mask,
            )[-1, :, :] # (batch, vocab_size)
            logits_list.append(logits)
            _, next_word = logits.max(1)
            next_word = next_word.view(1, 3)
            preds = torch.cat([preds, next_word], dim=0) # (target_len, batch)
            # TODO 全てがEOSだった場合終了
        logits = torch.stack(logits_list[1:], dim=0) # (target_len, batch, vocab_size)
    else:
        raise Exception(f"feedback must be 'teacher' or 'student', not {feedback}.")

    targets = targets[1:,:]
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
    
    if return_words:
        target_words = targets # (199, b)
        _, words = logits.max(2)
        words = words.detach() # (199, b)
        return loss, words, target_words
    else:
        return loss


def eval(
    gpu_id,
    seq2seq_model,
    seen_dataloader,
    unseen_dataloader,
    writer,
    now_epoch,
):
    seq2seq_model.eval()
    s = time.time()
    seen_losses = []
    for inputs, targets in seen_dataloader:
        loss = rollout(seq2seq_model, inputs, targets)
        seen_losses.append(loss.item())
    unseen_losses = []
    for inputs, targets in unseen_dataloader:
        loss = rollout(seq2seq_model, inputs, targets)
        unseen_losses.append(loss.item())
    seen_loss = torch.from_numpy(np.array([np.mean(seen_losses)])).to(gpu_id)
    all_reduce(seen_loss)
    unseen_loss = torch.from_numpy(np.array([np.mean(unseen_losses)])).to(gpu_id)
    all_reduce(unseen_loss)
    if int(os.environ["LOCAL_RANK"]) == 0:
        seen_loss = seen_loss.item() / int(os.environ["NP"])
        unseen_loss = unseen_loss.item() / int(os.environ["NP"])
        writer.add_scalar("eval/seen_loss", seen_loss, now_epoch)
        writer.add_scalar("eval/unseen_loss", unseen_loss, now_epoch)
        logger.info(f"========== Evaluation ==========")
        logger.info(f"Seen Loss:  {seen_loss:.5f}")
        logger.info(f"Unseen Loss:{unseen_loss:.5f}")
        logger.info(f"time:{((time.time() - s) / 60):.2f} [min]")
        logger.info(f"================================")


def train_one_step(
    seq2seq_model,
    seq2seq_optimizer,
    inputs,
    targets,
):
    seq2seq_model.train()
    seq2seq_optimizer.zero_grad()
    loss = rollout(seq2seq_model, inputs, targets)
    loss.backward()
    seq2seq_optimizer.step()
    return loss


def train(
    gpu_id: int,
    logger: logging.Logger,
    seq2seq_model: torch.nn.Module,
    model_name: str,
    train_data_path: str,
    train_data_num: int,
    val_seen_data_path: str,
    val_seen_data_num: int,
    val_unseen_data_path: str,
    val_unseen_data_num: int,
    use_image_feature: bool,
    dataset_drop_last: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    skip_frame_per: int,
    max_instruction_length: int,
    n_iters: int,
    save_every: int,
    log_every: int,
    eval_every: int,
):
    seq2seq_optimizer = optim.Adam(
        filter_param(seq2seq_model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    seq2seq_model.train()
    
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info("LOADING DATA...")
    train_dataset = R2RDataset(train_data_path, use_image_feature, train_data_num, True, skip_frame_per, max_instruction_length)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=True,
        rank=gpu_id,
    )
    train_dataloader = DataLoader(
        train_dataset,
        # num_workers=2,
        batch_size=batch_size,
        # shuffle=dataset_shuffle,
        drop_last=dataset_drop_last,
        collate_fn=my_collate_fn,
        sampler=train_sampler,
        # pin_memory=True,
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"The number of train data: {len(train_dataset)}, batch: {len(train_dataloader)}")
    val_seen_dataset = R2RDataset(val_seen_data_path, use_image_feature, val_seen_data_num, True, skip_frame_per, max_instruction_length)
    val_seen_sampler = DistributedSampler(
        val_seen_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=False,
        rank=gpu_id,
    )
    val_seen_dataloader = DataLoader(
        val_seen_dataset,
        # num_workers=2,
        batch_size=int(batch_size/4),
        # shuffle=False,
        drop_last=False,
        collate_fn=my_collate_fn,
        sampler=val_seen_sampler,
        # pin_memory=True,
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"The number of val_seen data: {len(val_seen_dataset)}, batch: {len(val_seen_dataloader)}")
    val_unseen_dataset = R2RDataset(val_unseen_data_path, use_image_feature, val_unseen_data_num, True, skip_frame_per, max_instruction_length)
    val_unseen_sampler = DistributedSampler(
        val_unseen_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=False,
        rank=gpu_id,
    )
    val_unseen_dataloader = DataLoader(
        val_unseen_dataset,
        # num_workers=2,
        batch_size=int(batch_size/4),
        # shuffle=False,
        drop_last=False,
        collate_fn=my_collate_fn,
        sampler=val_unseen_sampler,
        # pin_memory=True,
    )
    tb_log_dir = f"./data/models/{model_name}/tb"
    os.makedirs(tb_log_dir, exist_ok=True)

    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"The number of val_unseen data: {len(val_unseen_dataset)}, batch: {len(val_unseen_dataloader)}")
        logger.info("FINISH LOADING DATA!")
        logger.info("START TRAINING...")
    s = time.time()

    with (
            SummaryWriter(log_dir=tb_log_dir)
            if int(os.environ["LOCAL_RANK"]) == 0
            else contextlib.suppress()
    ) as writer:
        for i in range(1, n_iters + 1):
            losses = []
            for _, (inputs, targets) in enumerate(train_dataloader):
                loss = train_one_step(
                    seq2seq_model,
                    seq2seq_optimizer,
                    inputs,
                    targets,
                )
                losses.append(loss.item())

            train_loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
            all_reduce(train_loss)
            train_loss = train_loss.item() / int(os.environ["NP"])

            if int(os.environ["LOCAL_RANK"]) == 0:
                writer.add_scalar("train/loss", train_loss, i)

            if i % save_every == 0:
                save_model(seq2seq_model, f"./data/models/{model_name}/data/{i}")
            if i == 1 or i % log_every == 0:

                if int(os.environ["LOCAL_RANK"]) == 0:
                    logger.info(f"---------- Iteration {i}/{n_iters} ----------")
                    logger.info(f"train loss:{train_loss:.5f}")
                    logger.info(f"time:{((time.time() - s) / 60):.2f} [min]")
                s = time.time()
            if i == 1 or i % eval_every == 0:
                eval(gpu_id, seq2seq_model, val_seen_dataloader, val_unseen_dataloader, writer, i)


def main(config, model_name, logger, gpu_id):
    lang = R2RLang(name="r2r_train")
    seq2seq_model = Seq2SeqTransformer(
        num_encoder_layers=config["model"]["num_encoder_layers"],   
        num_decoder_layers=config["model"]["num_decoder_layers"],
        emb_size=config["model"]["emb_size"],
        vocab_emb_size=config["model"]["vocab_embedding_size"],
        nhead=config["model"]["nhead"],
        use_image_feature=config["train"]["use_image_feature"],
        vocab_size=lang.vocab_size,
        glove=lang.glove_vec,
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout_ratio"],
    )
    seq2seq_model = DistributedDataParallel(seq2seq_model.to(gpu_id), device_ids=[gpu_id])

    train(
        gpu_id=gpu_id,
        logger=logger,
        seq2seq_model=seq2seq_model,
        model_name=model_name,
        train_data_path=config["train"]["train_data_path"],
        train_data_num=config["train"]["train_data_num"],
        val_seen_data_path=config["train"]["val_seen_data_path"],
        val_seen_data_num=config["train"]["val_seen_data_num"],
        val_unseen_data_path=config["train"]["val_unseen_data_path"],
        val_unseen_data_num=config["train"]["val_unseen_data_num"],
        use_image_feature=config["train"]["use_image_feature"],
        dataset_drop_last=config["train"]["dataset_drop_last"],
        batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        skip_frame_per=config["train"]["skip_frame_per"],
        max_instruction_length=config["train"]["max_instruction_length"],
        n_iters=config["train"]["n_iters"],
        save_every=config["train"]["save_every"],
        log_every=config["train"]["log_every"],
        eval_every=config["train"]["eval_every"],
    )


if __name__=="__main__":
    f = open("debug.txt", "w")
    f.write(f"START transformer speaker!\n")
    f.close()
    rank = int(os.environ["LOCAL_RANK"])
    world_size = torch.cuda.device_count()
    n_proc = int(os.environ["NP"])
    gpu_id = rank % world_size
    torch.cuda.set_device(gpu_id)
    torch.distributed.init_process_group(backend="GLOO", init_method="env://", world_size=n_proc)
    print(f"rank: {rank}, world_size: {world_size}, gpu_id: {gpu_id}, n_proc: {n_proc}\n")
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', help='the path to config.')
    parser.add_argument('--model-name', help='the name of model to train.')
    args = parser.parse_args()
    
    with open(args.config_path, "r") as yml:
        config = yaml.safe_load(yml)
    
    os.makedirs(f"data/models/{args.model_name}", exist_ok=True)
    
    with open(f"data/models/{args.model_name}/config.yaml", "w") as f:
        yaml.dump(config, f)
        
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"data/models/{args.model_name}/train.log")
    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(levelname)s] %(asctime)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    if rank == 0:
        logger.info("Start!")
    
    main(config, args.model_name, logger, gpu_id)
