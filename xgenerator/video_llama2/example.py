"""
reference: https://github.com/DAMO-NLP-SG/VideoLLaMA2?tab=readme-ov-file#-inference
"""
import sys
from typing import List

import torch
import transformers
import numpy as np
from huggingface_hub import login

sys.path.append("/home/4/ud02274/navigation/VideoLLaMA2")
sys.path.append("/home/4/ud02274/navigation/myss")

from videollama2.conversation import conv_templates
from videollama2.constants import DEFAULT_MMODAL_TOKEN, MMODAL_TOKEN_INDEX
from videollama2.mm_utils import get_model_name_from_path, tokenizer_MMODAL_token, process_video, process_image
from videollama2.model.builder import load_pretrained_model
from access_tokens import HUGGINGFACE_ACCESS_TOKEN

login(token=HUGGINGFACE_ACCESS_TOKEN)


def inference(
    model_path: str,
    paths: List[str],
    questions: List[str],
    num_frames: int,
):
    model_name = get_model_name_from_path(model_path)
    print(f"model_name: {model_name}")
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda:0')

    tensor = process_video(
        paths[0],
        processor,
        model.config.image_aspect_ratio,
        num_frames,
    ).to(
        dtype=torch.float16,
        device='cuda',
        non_blocking=True,
    )
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"] # "<video>"
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"] # -201

    tensor = [tensor] # tensor: (8, 3, 336, 336)

    prompt = "[INST] <<SYS>>\n" \
            "A chat between a curious user and an artificial intelligence assistant." \
            "The assistant gives helpful, detailed, and polite answers to the user's questions." \
            "\n<</SYS>>\n\n <video>\nDescribe how the camera wearer moves around the indoor environment in 40 words or less. Follow the format of the output as shown in the example below.\n\n[Example of output format]\nTurn left and go down the steps on the left. Turn right and wait near the unicycle.\n[/Example of output format]\n[/INST]"

    input_ids = tokenizer_MMODAL_token(
        prompt, tokenizer, modal_token_index, return_tensors='pt',
    ).unsqueeze(0).to('cuda:0') # (1, instr_len)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images_or_videos=tensor,
            modal_list=['video'],
            do_sample=True,
            temperature=0.2,
            # max_new_tokens=1024,
            max_new_tokens=80,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    print(outputs[0])


if __name__ == "__main__":
    inference(
        # model_path='DAMO-NLP-SG/VideoLLaMA2-7B',   # 8 frames
        # model_path='DAMO-NLP-SG/VideoLLaMA2-7B-Base',   # 8 frames
        # model_path='DAMO-NLP-SG/VideoLLaMA2-7B-16F', # 16 frames
        # model_path='DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base', # 16 frames
        model_path="./data/models/video_llama2/finetune_videollama2_vllava_qlora_r128_a256_lr2en5/",
        paths=['/home/4/ud02274/navigation/myss/sound-spaces/data/models/ss1-savi/mp3d/savi-2nd/past-eprl-replaybuffer/beam1-k10-p0.95-t2.0/image_seqs_0.mp4'],
        questions=['Describe how the camera wearer moves around the indoor environment in 40 words or less. Follow the format of the output as shown in the example below.\n\n[Example of output format]\nTurn left and go down the steps on the left. Turn right and wait near the unicycle.\n[/Example of output format]\n'],
        num_frames=16,
    )
