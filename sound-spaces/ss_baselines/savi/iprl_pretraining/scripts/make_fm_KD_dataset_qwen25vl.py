import sys

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import numpy as np
from tqdm import trange
import lmdb
import msgpack_numpy
import torch
from PIL import Image

sys.path.append("/home/4/ud02274/navigation/myss")
sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")

from ss_baselines.savi.config.default import get_config
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_lmdb_dataset import IPRLPretrainingLMDBDataset


class DatasetConfig:
    model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    device: str = "cuda:0"
    
    # Train
    lmdb_dataset_path: str = "./data/lmdb_dataset/iprl_pretrain/train_cr02"
    split: str = "train_cr_0.2_w_past_instruction"
    data_num: int = 100398
    save_path: str = "./data/lmdb_dataset/iprl_pretrain/train_cr02_qwen25vl"

    # Val
    # lmdb_dataset_path: str = "./data/lmdb_dataset/iprl_pretrain/val"
    # split: str = "val_w_past_instruction"
    # data_num: int = 500
    # save_path: str = "./data/lmdb_dataset/iprl_pretrain/val_qwen25vl"

    text: str = "Describe how the camera wearer moves around the indoor environment in 40 words or less. Follow the format of the output as shown in the example below.\n\n[Example of output format]\nTurn left and go down the steps on the left. Turn right and wait near the unicycle.\n[/Example of output format]\n"

    start_idx: int = 0


def main(config: DatasetConfig):

    # Load a model and a processor
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # TODO installにめちゃ時間かかる...
        device_map=config.device,
    )
    processor = AutoProcessor.from_pretrained(config.model_path)

    task_config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml").TASK_CONFIG
    dataset = IPRLPretrainingLMDBDataset(
        config=task_config,
        split=config.split,
        lmdb_dataset_path=config.lmdb_dataset_path,
        data_num=config.data_num,
        foundation_model_type=None,
        fm_lmdb_dataset_path=None,
        environment_type="ss1-savi",
        device=config.device,
    )

    map_size = 15 * 1.1e12 # about 15 TB
    env = lmdb.open(config.save_path, map_size=int(map_size))

    for i in trange(config.start_idx, len(dataset)):
        x, _ = dataset[i]
        frames = (x["image_seq"].squeeze(1)[:, :, :, :3] * 255).astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]

        # Prepare inputs
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": frames,
                    },
                    {"type": "text", "text": config.text},
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # fps=1.0,
            padding=True,
            return_tensors="pt",
            **video_kwargs,
        )
        inputs = inputs.to(config.device)

        # Generate outputs
        outputs = model.generate(
            **inputs,
            # do_sample=True,
            # temperature=0.2,
            max_new_tokens=80,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Calc logits
        logits = torch.stack(outputs.scores)[:, 0, :]

        # Visualization
        # generated_ids = [torch.argmax(logits, axis=1)]
        # input_text = processor.batch_decode(
        #     inputs.input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        # )
        # output_text = processor.batch_decode(
        #     generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        # )
        # print(f"input: {input_text}")
        # print(f"output: {output_text}")

        data = [
            logits.to('cpu').detach().numpy().copy(),
        ]

        with env.begin(write=True) as txn:
            txn.put(
                f"{i}".encode(), 
                msgpack_numpy.packb(
                    data, use_bin_type=True
                ),
            )

if __name__=="__main__":
    main(
        config=DatasetConfig()
    )