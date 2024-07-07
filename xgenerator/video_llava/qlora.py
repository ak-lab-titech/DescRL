import sys
import os
import logging
import argparse

import yaml
from transformers import Trainer, AutoProcessor, DataCollatorForSeq2Seq, VideoLlavaProcessor, TrainingArguments, VideoLlavaForConditionalGeneration, BitsAndBytesConfig
import numpy as np
import torch
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dotmap import DotMap

sys.path.append("/home/4/ud02274/navigation/myss")

from soundspaces.utils import generate_video
from xgenerator.common.hugging_face_utils import HFR2RDataset, find_all_linear_names
from xgenerator.common.utils import MyLogger


MODEL_NAME = "LanguageBind/Video-LLaVA-7B-hf"
PROCESSOR = AutoProcessor.from_pretrained(MODEL_NAME)
PROCESSOR.tokenizer.padding_side = "right"


def video_llava_collate_fn(batch):
    videos = []
    targets = []
    prompts = []
    
    for i, (x, y) in enumerate(batch):
        videos.append(x)
        prompts.append(
            f"USER: <video>What is the camera wearer doing? ASSISTANT: {y}"
        )

    inputs = PROCESSOR(
        videos=videos,
        text=prompts,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    labels = inputs["input_ids"]
    labels[labels == PROCESSOR.tokenizer.pad_token_id] = -100 # To ignore the part of padding, when calculating loss.
    inputs["labels"] = labels

    return inputs


class VideoLlaVADataCollatorForVideoSeq2Seq(DataCollatorForSeq2Seq):
    def __init__(self, tokenizer, max_instruction_length, pad_to_multiple_of):
        super().__init__(tokenizer, pad_to_multiple_of)
        self.max_instruction_length = max_instruction_length
    
    def __call__(self, batch, return_tensors=None):
        inputs = video_llava_collate_fn(batch)
        return inputs


def main(config, logger):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = HFR2RDataset(
        data_path=config.train.data_path,
        data_num=config.train.data_num,
        max_instruction_length=config.train.max_instruction_length,
        n_slice=8,
    )
    val_dataset = HFR2RDataset(
        data_path=config.val.data_path,
        data_num=config.val.data_num,
        max_instruction_length=config.val.max_instruction_length,
        n_slice=8,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = VideoLlavaForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
        device_map="auto",
    )

    logger.info(f"config: {model.config}")

    # for k, v in model.named_parameters():
    #     if "vision_model" in k or "language_model" in k:
    #         v.requires_grad = False
    # model.enable_input_require_grads()

    peft_config = LoraConfig(
        r=config.train.lora.r,
        lora_alpha=config.train.lora.lora_alpha,
        lora_dropout=config.train.lora.lora_dropout,
        init_lora_weights=config.train.lora.init_lora_weights,
        inference_mode=False,
        target_modules=find_all_linear_names(model),
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=f"{config.save_path}/models",          # output directory
        num_train_epochs=config.train.num_epochs,              # total number of training epochs
        gradient_accumulation_steps=config.train.accum_iter,
        learning_rate=config.train.learning_rate,
        per_device_train_batch_size=config.train.batch_size,  # batch size per device during training
        per_device_eval_batch_size=config.train.batch_size,   # batch size for evaluation
        warmup_steps=config.train.warmup_steps,                # number of warmup steps for learning rate scheduler
        # weight_decay=0.001,               # strength of weight decay
        logging_dir=f"{config.save_path}/tb",            # directory for storing logs
        logging_steps=config.train.logging_steps,
        fp16=True,
    )

    # training_args.load_best_model_at_end = True

    logger.info(f"training_args")
    logger.info(training_args)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=VideoLlaVADataCollatorForVideoSeq2Seq(
            PROCESSOR.tokenizer,
            max_instruction_length=config.train.max_instruction_length,
            pad_to_multiple_of=8 if training_args.fp16 or training_args.bf16 else None,
        ),
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

    model.save_pretrained(training_args.output_dir)
    PROCESSOR.save_pretrained(training_args.output_dir)

    logger.info(f"Evaluation")
    for video, gt_instr in val_dataset:
        inputs = PROCESSOR(
            videos=video,
            text="USER: <video>What is the camera wearer doing? ASSISTANT: ",
            return_tensors="pt",
        )
        for k, v in inputs.items():
            inputs[k] = v.cuda()
        out = model.generate(max_new_tokens=40, **inputs)
        pred_instr = PROCESSOR.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        logger.info(f"-------")
        logger.info(f"True: {gt_instr}")
        logger.info(f"Pred: {pred_instr}")


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='the name of model to train.')
    args = parser.parse_args()

    with open("/home/4/ud02274/navigation/myss/xgenerator/video_llava/qlora_config.yaml", "r") as yml:
        config = yaml.safe_load(yml)
    
    config["save_path"] = f"data/models/video_llava/{args.model_name}"
    
    os.makedirs(config['save_path'], exist_ok=True)
    with open(f"{config['save_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)
    
    config = DotMap(config)
    logger = MyLogger("logger", logging.INFO, f"{config.save_path}/experiment.log")

    main(config, logger)
