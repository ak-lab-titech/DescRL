"""
reference: https://github.com/DAMO-NLP-SG/VideoLLaMA2/blob/main/videollama2/train.py
"""

import os
import sys
import pathlib
from typing import Dict, Any, Sequence, Tuple
import logging
import argparse

import yaml
import torch
import numpy as np
import transformers
import deepspeed
from dotmap import DotMap
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from peft.tuners.lora import LoraLayer
from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock
from huggingface_hub import login

sys.path.append("/home/4/ud02274/navigation/VideoLLaMA2")
sys.path.append("/home/4/ud02274/navigation/myss")

from videollama2 import conversation as conversation_lib
from videollama2.model import Videollama2LlamaForCausalLM, Videollama2MistralForCausalLM, Videollama2MixtralForCausalLM
from videollama2.constants import NUM_FRAMES, IGNORE_INDEX
from videollama2.videollama2_trainer import (
    VideoLLaMA2Trainer,
    get_peft_state_maybe_zero_3, get_peft_state_non_lora_maybe_zero_3, 
    find_all_linear_names, safe_save_model_for_hf_trainer
)
from videollama2.train import (
    ModelArguments,
    TrainingArguments,
    smart_tokenizer_and_embedding_resize,
    preprocess,
    preprocess_multimodal,
    process_video,
)
from xgenerator.common.utils import set_seed, MyLogger
from xgenerator.common.hugging_face_utils import HFR2RDataset
from access_tokens import HUGGINGFACE_ACCESS_TOKEN

login(token=HUGGINGFACE_ACCESS_TOKEN)


class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""
    
    def __init__(
        self,
        tokenizer,
        data_args,
    ):
        self.tokenizer = tokenizer
        self.data_args = data_args

    def __call__(self, instances: Sequence[Tuple[torch.Tensor, str]]) -> Dict[str, torch.Tensor]:
        # TODO audio未対応

        videos = []
        labels = []
        keys = []
        input_ids = []
        for instance in instances:
            video = instance[0].numpy().copy() # (l, h, w, c)
            video = process_video(
                video,
                self.data_args.video_processor,
                self.data_args.image_aspect_ratio,
                self.data_args.num_frames,
            ) # (l, c, h, w), max: 2.1458969116210938, min: -1.7922625541687012
            videos.append(video)
            keys.append("video")

            conservations = [[
                {
                    "from": "human",
                    "value": f"<video>\nDescribe how the camera wearer moves around the indoor environment in 40 words or less. Follow the format of the output as shown in the example below.\n\n[Example of output format]\nTurn left and go down the steps on the left. Turn right and wait near the unicycle.\n[/Example of output format]\n",
                },
                {
                    "from": "gpt",
                    "value": f"{instance[1]}",
                },
            ]]
            conservations = preprocess_multimodal(
                conservations, 
                self.data_args,
            )
            conservations = preprocess(
                conservations,
                self.tokenizer,
                MODAL_list=['VIDEO'],
            )
            input_id = conservations["input_ids"][0]
            label = conservations["labels"][0]
            input_ids.append(input_id)
            labels.append(label)

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )[:self.tokenizer.model_max_length]

        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )[:self.tokenizer.model_max_length]

        batch = {
            "images": [videos, keys],
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = HFR2RDataset(
        data_path=data_args.data_path,
        data_num=data_args.data_num,
        max_instruction_length=data_args.max_instruction_length,
        n_slice=data_args.num_frames,
        need_action_and_depth_semantic=False,
    )
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )


def main(
    config: Dict[str, Any],
    logger: logging.Logger,
    attn_implementation: str=None,
):
    set_seed(config["seed"])

    model_args = ModelArguments(**config["model"])
    data_args = DotMap(config["data"])
    training_args = TrainingArguments(**config["train"])

    logger.info("------------ model_args ------------")
    logger.info(model_args)
    logger.info("------------ data_args ------------")
    logger.info(data_args)
    logger.info("------------ training_args ------------")
    logger.info(training_args)

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            # device_map={"": training_args.device},
            # load_in_4bit=training_args.bits == 4,
            # load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type, # {'fp4', 'nf4'}
                bnb_4bit_quant_storage=compute_dtype,
            )
        ))

    
    if model_args.pretrain_model_name_or_path is not None:
        assert os.path.exists(model_args.pretrain_model_name_or_path)
        pretrain_model_name_or_path = model_args.pretrain_model_name_or_path
    else:
        pretrain_model_name_or_path = model_args.model_name_or_path
    
    if model_args.vision_tower is not None:
        if 'vicuna' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2LlamaForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'mistral' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MistralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
        elif 'mixtral' in model_args.model_name_or_path.lower():
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MixtralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
            deepspeed.utils.set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
        else:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config._attn_implementation = attn_implementation
            model = Videollama2MistralForCausalLM.from_pretrained(
                pretrain_model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                do_sample=True,
                **bnb_model_from_pretrained_args
            )
    else:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config._attn_implementation = attn_implementation
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrain_model_name_or_path,
            config=config,
            cache_dir=training_args.cache_dir,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            do_sample=True,
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        logger.info("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
    )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            if model_args.version == "v1":
                conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
            elif model_args.version == "v1_mistral":
                conversation_lib.default_conversation = conversation_lib.conv_templates["mistral_instruct"]

    if model_args.vision_tower is not None:
        # initialize vision encoder + multi-modal projector
        model.get_model().initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)

        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.video_processor = vision_tower.video_processor if hasattr(vision_tower, "video_processor") else vision_tower.image_processor

        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_MM_tokenizer(model_args, tokenizer=tokenizer)

        model.config.num_frames = data_args.num_frames

        logging.info(f"frames: {model.config.num_frames}")

    if training_args.bits in [4, 8]:
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    logging.info("Current model:", model)
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    logging.info(f"data_module: {data_module}")

    # select a Trainer
    trainer = VideoLLaMA2Trainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(model.named_parameters(), training_args.lora_bias)
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(model.named_parameters())
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            tokenizer.save_pretrained(training_args.output_dir)
            # data_args.video_processor.save_pretrained(training_args.output_dir)
    else:
        safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='the name of model to train.')
    args = parser.parse_args()

    with open("/home/4/ud02274/navigation/myss/xgenerator/video_llama2/qlora_config.yaml", "r") as yml:
        config = yaml.safe_load(yml)
    
    config["save_path"] = f"data/models/video_llama2/{args.model_name}"
    config["train"]["output_dir"] = f"{config['save_path']}"
    config["train"]["logging_dir"] = f"{config['save_path']}/tb"
    
    os.makedirs(config['save_path'], exist_ok=True)
    with open(f"{config['save_path']}/config.yaml", "w") as f:
        yaml.dump(config, f)
    
    logger = MyLogger("logger", logging.INFO, f"{config['save_path']}/experiment.log")

    main(config, logger, attn_implementation="flash_attention_2")
