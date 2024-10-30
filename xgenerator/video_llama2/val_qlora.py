import sys

import yaml
import numpy as np
import torch
from dotmap import DotMap

sys.path.append("/home/4/ud02274/navigation/VideoLLaMA2")
sys.path.append("/home/4/ud02274/navigation/myss")

from xgenerator.video_llama2.qlora import make_supervised_data_module
from videollama2.mm_utils import get_model_name_from_path
from videollama2.model.builder import load_pretrained_model
from videollama2.train import process_video
from videollama2.constants import MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token


def main(model_path, data_args):
    tokenizer, model, processor, _ = load_pretrained_model(model_path, None, get_model_name_from_path(model_path))
    data_args.video_processor = processor
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)

    dataset = data_module["train_dataset"]

    for i, instance in enumerate(dataset):
        print(f"--- True (i = {i}) ---")
        print(instance[1])
        
        prompt = "[INST] <<SYS>>\n" \
                "A chat between a curious user and an artificial intelligence assistant." \
                "The assistant gives helpful, detailed, and polite answers to the user's questions." \
                "\n<</SYS>>\n\n <video>\nDescribe how the camera wearer moves around the indoor environment in 40 words or less. Follow the format of the output as shown in the example below.\n\n[Example of output format]\nTurn left and go down the steps on the left. Turn right and wait near the unicycle.\n[/Example of output format]\n[/INST]"
        input_ids = tokenizer_MMODAL_token(
            prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors='pt',
        ).unsqueeze(0).to(device="cuda")
        tensor = process_video(
            instance[0].numpy().copy(),
            data_args.video_processor,
            data_args.image_aspect_ratio,
            data_args.num_frames,
        ).to(
            dtype=torch.float16,
            device='cuda',
            non_blocking=True,
        )
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images_or_videos=[tensor],
                modal_list=['video'],
                do_sample=True,
                temperature=0.2,
                # max_new_tokens=1024,
                max_new_tokens=80,
                use_cache=True,
            )
        pred_sentence = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
        print(f"--- Pred (i = {i}) ----")
        print(pred_sentence)



if __name__=="__main__":
    model_path = "./data/models/video_llama2/finetune_videollama2_vllava_qlora_r128_a256_lr2en5"
    data_path = "/home/4/ud02274/navigation/my-VLN-CE/data/trajectories_dirs/xgen_pretraining/val_unseen_trajectories.lmdb"
    data_num = 10


    with open(f"{model_path}/config.yaml", "r") as yml:
        config = yaml.safe_load(yml)

    data_args = DotMap(config["data"])
    data_args.data_path = data_path
    data_args.data_num = data_num

    main(model_path, data_args)
