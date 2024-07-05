import av
import torch
import numpy as np
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor, BitsAndBytesConfig


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])



def main(
    model_path: str,
    video_path: str,
    prompt: str,
    max_new_tokens: int,
) -> None:
    model = VideoLlavaForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    processor = VideoLlavaProcessor.from_pretrained(model_path)

    # prepair a video
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames
    indices = np.arange(0, total_frames, total_frames / 8).astype(int)
    video = read_video_pyav(container, indices)
    print(f"video: {type(video)}, shape: {np.shape(video)}, max: {np.max(video)}, min: {np.min(video)}")

    inputs = processor(text=prompt, videos=video, return_tensors="pt")

    for k, v in inputs.items():
        if torch.cuda.is_available():
            inputs[k] = inputs[k].to("cuda")
        
        print(f"{k}: {np.shape(v)}")

    out = model.generate(max_new_tokens=max_new_tokens, **inputs)
    print(f"out: {out}")
    out_text = processor.batch_decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    print(f"out_text[0]: {out_text[0]}")
    print(f"out_text[len(prompt)-7:]: {out_text[0][len(prompt)-7:]}") # minus len("<video>") (=7)


if __name__=="__main__":
    model_path = "LanguageBind/Video-LLaVA-7B-hf"
    # model_path = "data/models/video-llava/lora-6"

    video_path = "data/video/finetune/original.mp4"
    # video_path = "./data/video/val_seen/image_seq_0.mp4"

    prompt = "USER: <video>What is the camera wearer doing? ASSISTANT:"

    main(
        model_path=model_path,
        video_path=video_path,
        prompt=prompt,
        max_new_tokens=40,
    )

