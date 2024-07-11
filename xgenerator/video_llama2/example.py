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
):
    model_name = get_model_name_from_path(model_path)
    print(f"model_name: {model_name}")
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, None, model_name)
    model = model.to('cuda:0')
    conv_mode = 'llama_2'

    tensor = process_video(
        paths[0],
        processor,
        model.config.image_aspect_ratio,
    ).to(
        dtype=torch.float16,
        device='cuda',
        non_blocking=True,
    )
    default_mm_token = DEFAULT_MMODAL_TOKEN["VIDEO"] # "<video>"
    modal_token_index = MMODAL_TOKEN_INDEX["VIDEO"] # -201

    tensor = [tensor] # tensor: (8, 3, 336, 336)


    question = default_mm_token + "\n" + questions[0]
    conv = conv_templates[conv_mode].copy()
    print(f"conv.roles[0]: {conv.roles[0]}")
    print(f"conv.roles[1]: {conv.roles[1]}")
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    """promptは以下のようになっている

    [INST] <<SYS>>
    You are a helpful, respectful and honest assistant. 
    Always answer as helpfully as possible, while being safe. 
    Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.

    If a question does not make any sense, or is not factually coherent,
    explain why instead of answering something not correct.
    If you don't know the answer to a question, please don't share false information.
    <</SYS>>

    <video>
    What is the camera wearer doing? [/INST]
    """

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
        model_path='DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base', # 16 frames
        paths=['data/video/finetune/original.mp4'],
        questions=['What is the camera wearer doing?'],
    )


"""
# data/finetune/original.mp4を使った時の結果

## DAMO-NLP-SG/VideoLLaMA2-7B
The camera wearer is walking through the house.


## DAMO-NLP-SG/VideoLLaMA2-7B-Base
The camera is being worn by someone who is walking through a room with
a wooden ceiling and blue walls. The room appears to be empty, and there
are no other people or objects visible. The person wearing the camera is not
doing anything specific, but they are simply moving through the space. The
video does not provide any context or information about the purpose of the
camera or the person wearing it.


## DAMO-NLP-SG/VideoLLaMA2-7B-16F
The camera wearer is walking. 


## DAMO-NLP-SG/VideoLLaMA2-7B-16F-Base

### max_new_tokens = 1024
The camera is being worn by someone as they explore an attic space 
with wooden beams and a staircase leading down. The space is dimly 
lit and has a cozy, rustic feel. The person wearing the camera appears
to be taking a tour of the space, possibly for the purpose of renovating
or decorating it. The video provides a first-person perspective of the attic,
allowing viewers to experience the space as if they were there in person.
Overall, the video creates a sense of curiosity and exploration, inviting
viewers to imagine what they might find in this hidden, overlooked area of a house.

### max_new_tokens = 80
The camera is being worn by someone as they explore a room with a high ceiling
and wooden beams. The room has a cozy and inviting atmosphere, with a comfortable
couch and a bookshelf. The person wearing the camera appears to be taking a tour
of the space, possibly for the purpose of filming or taking photographs. The room is
well-lit, with natural light coming in

### max_new_tokens = 40
The camera is being worn by someone as they explore a room with a high ceiling,
wooden beams, and a blue carpet. The room appears to be unoccupied and has a
few pieces of


"""
