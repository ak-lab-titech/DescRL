import time

import av
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoTokenizer, VisionEncoderDecoderModel


def make_video_from_list(l, output_file, frame_rate):
    frame_size = (np.shape(l)[1], np.shape(l)[2])
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, frame_rate, frame_size)

    for i in range(len(l)):
        frame = l[i]
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

device = "cuda" if torch.cuda.is_available() else "cpu"

# load pretrained processor, tokenizer, and model
image_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = VisionEncoderDecoderModel.from_pretrained("Neleac/timesformer-gpt2-video-captioning").to(device)

# load video
video_path = "./data/sample_video/episode_0.mp4"
container = av.open(video_path)

# extract evenly spaced frames from video
# seg_len = container.streams.video[0].frames
# clip_len = model.config.encoder.num_frames
# indices = set(np.linspace(0, seg_len, num=clip_len, endpoint=False).astype(np.int64))
frames = []
container.seek(0)
for i, frame in enumerate(container.decode(video=0)):
    # if i in indices:
    frames.append(frame.to_ndarray(format="rgb24"))

# generate caption
gen_kwargs = {
    "min_length": 30, 
    "max_length": 40,
    "num_beams": 1,
}
print("START")
s = time.time()
pixel_values = image_processor(frames, return_tensors="pt").pixel_values.to(device)
print(type(frames))
print(np.shape(frames))
print(f"pixel values: {time.time() - s}")
s = time.time()
tokens = model.generate(pixel_values, **gen_kwargs)
print(f"tokens: {time.time() - s}")
s = time.time()
caption = tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]
print(f"caption: {time.time() - s}")
print(caption)

make_video_from_list(frames, "hoge.mp4", 5)