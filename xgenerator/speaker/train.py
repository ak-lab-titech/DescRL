import os
import sys
import time
import argparse
import logging

import yaml
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.distributions as D
from torch.utils.data import DataLoader

sys.path.append("/home/0/19B30511/av-nav/myss/xgenerator")

from common.utils import try_cuda
from common.lang import R2RLang
from common.load_lmdb import R2RDataset, my_collate_fn
from model import SpeakerEncoderLSTM, SpeakerDecoderLSTM

EOS_IDX = 0
BOS_IDX = 1 # TODO 要修正。元にしているgloveから取ってきた方が良さそう


def filter_param(param_list):
    return [p for p in param_list if p.requires_grad]


def save_model(encoder, decoder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    encoder_path = f"{save_dir}/encoder.pth"
    decoder_path = f"{save_dir}/decoder.pth"
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)

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
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    action_embeddings = inputs["action_seqs"]
    image_features = inputs["image_seqs"]
    path_mask = inputs["mask"]
    batch_size = image_features[0].shape[0]
    
    ctx, h_t, c_t = encoder(action_embeddings, image_features)
    
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
        
        all_seq_end = (w_t == 0).all()
        if all_seq_end:
            break
        
        log_probs = F.log_softmax(logit, dim=1)
        loss += F.nll_loss(
            log_probs,
            target,
            ignore_index=EOS_IDX,
            reduction="mean",
        )
    
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    
    return loss


def train(
    logger: logging.Logger,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    model_name: str,
    data_path: str,
    use_image_feature: bool,
    dataset_shuffle: bool,
    dataset_drop_last: bool,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    max_instruction_length: int,
    feedback: str,
    n_iters: int,
    save_every: int,
    log_every: int,
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
    
    logger.info("LOADING DATA...")
    dataset = R2RDataset(data_path, use_image_feature)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=dataset_shuffle,
        drop_last=dataset_drop_last,
        collate_fn=my_collate_fn,
    )
    logger.info("FINISH LOADING DATA!")
    logger.info("START TRAINING...")
    s = time.time()
    for i in range(1, n_iters + 1):
        cnt = 0
        loss = 0
        for j, (inputs, targets) in enumerate(dataloader):
            loss_j = train_one_step(
                encoder,
                decoder,
                encoder_optimizer,
                decoder_optimizer,
                inputs,
                targets,
                max_instruction_length,
                feedback,
            )
            loss += loss_j
            cnt += 1
        if i % save_every == 0:
            save_model(encoder, decoder, f"./data/models/{model_name}/{i}")
        if i == 1 or i % log_every == 0:
            logger.info(f"---------- Iteration {i}/{n_iters} ----------")
            logger.info(f"loss:{loss/cnt:.5f}")
            logger.info(f"time:{((time.time() - s) / 60):.2f} [min]")
            s = time.time()


def main(config, model_name, logger):
    lang = R2RLang(name="r2r_train")
    encoder = SpeakerEncoderLSTM(
        action_embedding_size=4,
        world_embedding_size=config["model"]["world_embedding_size"],
        hidden_size=config["model"]["hidden_size"],
        dropout_ratio=config["model"]["dropout_ratio"],
        bidirectional=config["model"]["bidirectional"],
    )
    decoder = SpeakerDecoderLSTM(
        vocab_size=lang.vocab_size,
        vocab_embedding_size=config["model"]["vocab_embedding_size"],
        hidden_size=config["model"]["hidden_size"],
        dropout_ratio=config["model"]["dropout_ratio"],
        glove=lang.glove_vec,
        use_input_att_feed=config["model"]["use_input_att_feed"],
    )
    train(
        logger=logger,
        encoder=encoder,
        decoder=decoder,
        model_name=model_name,
        data_path=config["train"]["data_path"],
        use_image_feature=config["train"]["use_image_feature"],
        dataset_shuffle=config["train"]["dataset_shuffle"],
        dataset_drop_last=config["train"]["dataset_drop_last"],
        batch_size=config["train"]["batch_size"],
        learning_rate=config["train"]["learning_rate"],
        weight_decay=config["train"]["weight_decay"],
        max_instruction_length=config["train"]["max_instruction_length"],
        feedback=config["train"]["feedback"],
        n_iters=config["train"]["n_iters"],
        save_every=config["train"]["save_every"],
        log_every=config["train"]["log_every"]
    )


if __name__=="__main__":
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
    logger.info("Start!")
    
    main(config, args.model_name, logger)
