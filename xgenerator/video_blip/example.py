import sys
import os


import torch
from transformers import Blip2Processor
import cv2
import av
import numpy as np
from peft import LoraConfig, get_peft_model, PeftModel

sys.path.append("/home/4/ud02274/navigation/myss")

from xgenerator.video_blip.model import VideoBlipForConditionalGeneration, process
from xgenerator.common.load_lmdb import R2RDataset
from soundspaces.utils import generate_video


def main(
    model_path: str,
    video_path: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Blip2Processor.from_pretrained(model_path)
    model = VideoBlipForConditionalGeneration.from_pretrained(model_path).to(device)

    frames = []
    container = av.open(video_path)
    container.seek(0)
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
    frames = torch.from_numpy(np.array(frames))
    frames = frames.view((1,) + np.shape(frames)).permute(0, 4, 1, 2, 3) # (b, c, t, h, w)

    inputs = process(processor, video=frames, text=prompt).to(device)

    for k, v in inputs.items():
        print(f"{k}: {np.shape(v)}")

    with torch.no_grad():
        generated_ids = model.generate(max_new_tokens=max_new_tokens, **inputs)
    generated_text = processor.batch_decode(
        generated_ids, skip_special_tokens=False
    )[0].strip()
    print(f"generated_text: {generated_text}")


if __name__=="__main__":
    # model_path = "kpyu/video-blip-flan-t5-xl-ego4d"
    model_path = "kpyu/video-blip-opt-2.7b-ego4d"

    # video_path = "data/video/val_seen/image_seq_0.mp4"
    video_path = "data/video/finetune/original.mp4"

    prompt = "Question: what is the camera wearer doing? Answer:"
    
    main(
        model_path=model_path,
        video_path=video_path,
        prompt=prompt,
        max_new_tokens=80,
    )

