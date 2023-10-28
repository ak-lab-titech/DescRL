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
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM


def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]


def save_model(encoder, decoder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = f"{save_dir}/encoder.pth"
    decoder_path = f"{save_dir}/decoder.pth"
    torch.save(encoder.module.state_dict(), encoder_path)
    torch.save(decoder.module.state_dict(), decoder_path)


def rollout(encoder, decoder, inputs, targets, max_instruction_length, feedback):
    action_embeddings = [
        try_cuda(Variable(
            torch.from_numpy(act),
            requires_grad=False,
        )) for act in inputs["action_seqs"]
    ]
    image_features = [
        try_cuda(Variable(
            torch.from_numpy(img),
            requires_grad=False,
        )) for img in inputs["image_seqs"]
    ]
    path_mask = try_cuda(torch.from_numpy(inputs["mask"]))
    targets = try_cuda(torch.from_numpy(np.array(targets)))
    
    batch_size = image_features[0].shape[0]
    
    ctx, h_t, c_t = encoder(action_embeddings, image_features, inputs["seq_lengths"])
    
    w_t = try_cuda(Variable(
        torch.from_numpy(
            np.full(
                (batch_size,),
                BOS_IDX,
                dtype='int64',
            )
        ).long(),
        requires_grad=False,
    ))
    
    loss = 0
    ended = np.array([False] * batch_size)
    for t in range(max_instruction_length):
        h_t, c_t, _, logit = decoder(
            w_t.view(-1, 1), h_t, c_t, ctx, path_mask
        )
        target = targets[:,t].contiguous()

        # Determine next model inputs
        if feedback == 'teacher':
            w_t = target
        elif feedback == 'argmax':
            _, w_t = logit.max(1)
            w_t = w_t.detach()
        elif feedback == 'sample':
            probs = F.softmax(logit)
            m = D.Categorical(probs)
            w_t = m.sample()
        else:
            sys.exit('Invalid feedback option')
        
        log_probs = F.log_softmax(logit, dim=1)
        loss += F.nll_loss(
            log_probs,
            target,
            ignore_index=PAD_IDX,
            reduction="mean",
        )

        tmp = target.to("cpu").detach().numpy().copy()
        ended[tmp == EOS_IDX] = True
        if ended.all():
            break
    
    return loss


def eval(
    gpu_id,
    encoder,
    decoder,
    seen_dataloader,
    unseen_dataloader,
    max_instruction_length,
    writer,
    now_epoch,
):
    encoder.eval()
    decoder.eval()
    s = time.time()
    seen_losses = []
    for inputs, targets in seen_dataloader:
        loss = rollout(encoder, decoder, inputs, targets, max_instruction_length, "argmax")
        seen_losses.append(loss.item())
    unseen_losses = []
    for inputs, targets in unseen_dataloader:
        loss = rollout(encoder, decoder, inputs, targets, max_instruction_length, "argmax")
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
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    inputs,
    targets,
    max_instruction_length,
    feedback,
):
    encoder.train()
    decoder.train()
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = rollout(encoder, decoder, inputs, targets, max_instruction_length, feedback)
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss


def train(
    gpu_id: int,
    logger: logging.Logger,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    model_name: str,
    train_data_path: str,
    val_seen_data_path: str,
    val_unseen_data_path: str,
    use_image_feature: bool,
    dataset_drop_last: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_instruction_length: int,
    feedback: str,
    n_iters: int,
    save_every: int,
    log_every: int,
    eval_every: int,
):
    encoder_optimizer = optim.Adam(
        filter_param(encoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    decoder_optimizer = optim.Adam(
        filter_param(decoder.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    encoder.train()
    decoder.train()
    
    if multiprocessing.get_start_method() == 'fork':
        multiprocessing.set_start_method('spawn', force=True)
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info("LOADING DATA...")
    train_dataset = R2RDataset(train_data_path, use_image_feature)
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=True,
        rank=gpu_id,
    )
    train_dataloader = DataLoader(
        train_dataset,
        num_workers=2,
        batch_size=batch_size,
        # shuffle=dataset_shuffle,
        drop_last=dataset_drop_last,
        collate_fn=my_collate_fn,
        sampler=train_sampler,
        pin_memory=True,
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"The number of train data: {len(train_dataset)}, batch: {len(train_dataloader)}")
    val_seen_dataset = R2RDataset(val_seen_data_path, use_image_feature)
    val_seen_sampler = DistributedSampler(
        val_seen_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=False,
        rank=gpu_id,
    )
    val_seen_dataloader = DataLoader(
        val_seen_dataset,
        num_workers=2,
        batch_size=batch_size,
        # shuffle=False,
        drop_last=False,
        collate_fn=my_collate_fn,
        sampler=val_seen_sampler,
        pin_memory=True,
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        logger.info(f"The number of val_seen data: {len(val_seen_dataset)}, batch: {len(val_seen_dataloader)}")
    val_unseen_dataset = R2RDataset(val_unseen_data_path, use_image_feature)
    val_unseen_sampler = DistributedSampler(
        val_unseen_dataset,
        num_replicas=int(os.environ["NP"]),
        shuffle=False,
        rank=gpu_id,
    )
    val_unseen_dataloader = DataLoader(
        val_unseen_dataset,
        num_workers=2,
        batch_size=batch_size,
        # shuffle=False,
        drop_last=False,
        collate_fn=my_collate_fn,
        sampler=val_unseen_sampler,
        pin_memory=True,
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
                    encoder,
                    decoder,
                    encoder_optimizer,
                    decoder_optimizer,
                    inputs,
                    targets,
                    max_instruction_length,
                    feedback,
                )
                losses.append(loss.item())

            train_loss = torch.from_numpy(np.array([np.mean(losses)])).to(gpu_id)
            all_reduce(train_loss)
            train_loss = train_loss.item() / int(os.environ["NP"])

            if int(os.environ["LOCAL_RANK"]) == 0:
                writer.add_scalar("train/loss", train_loss, i)

            if i % save_every == 0:
                save_model(encoder, decoder, f"./data/models/{model_name}/{i}")
            if i == 1 or i % log_every == 0:

                if int(os.environ["LOCAL_RANK"]) == 0:
                    logger.info(f"---------- Iteration {i}/{n_iters} ----------")
                    logger.info(f"train loss:{train_loss:.5f}")
                    logger.info(f"time:{((time.time() - s) / 60):.2f} [min]")
                s = time.time()
            if i == 1 or i % eval_every == 0:
                eval(gpu_id, encoder, decoder, val_seen_dataloader, val_unseen_dataloader, max_instruction_length, writer, i)


def main(config, model_name, logger, gpu_id):
    lang = R2RLang(name="r2r_train")
    encoder = SpeakerEncoderLSTM(
        action_embedding_size=4,
        world_embedding_size=config["model"]["world_embedding_size"],
        hidden_size=config["model"]["hidden_size"],
        dropout_ratio=config["model"]["dropout_ratio"],
        bidirectional=config["model"]["bidirectional"],
    )
    encoder = DistributedDataParallel(encoder.to(gpu_id), device_ids=[gpu_id])
    decoder = SpeakerDecoderLSTM(
        vocab_size=lang.vocab_size,
        vocab_embedding_size=config["model"]["vocab_embedding_size"],
        hidden_size=config["model"]["hidden_size"],
        dropout_ratio=config["model"]["dropout_ratio"],
        glove=lang.glove_vec,
        use_input_att_feed=config["model"]["use_input_att_feed"],
    )
    decoder = DistributedDataParallel(decoder.to(gpu_id), device_ids=[gpu_id])
    train(
        gpu_id=gpu_id,
        logger=logger,
        encoder=encoder,
        decoder=decoder,
        model_name=model_name,
        train_data_path=config["train"]["train_data_path"],
        val_seen_data_path=config["train"]["val_seen_data_path"],
        val_unseen_data_path=config["train"]["val_unseen_data_path"],
        use_image_feature=config["train"]["use_image_feature"],
        dataset_drop_last=config["train"]["dataset_drop_last"],
        batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        max_instruction_length=config["train"]["max_instruction_length"],
        feedback=config["train"]["feedback"],
        n_iters=config["train"]["n_iters"],
        save_every=config["train"]["save_every"],
        log_every=config["train"]["log_every"],
        eval_every=config["train"]["eval_every"],
    )


if __name__=="__main__":
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
