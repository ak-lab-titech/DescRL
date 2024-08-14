"""
Make a dataset for knowledge distillation from a foundation model.
"""
import sys

import numpy as np
from tqdm import trange
import lmdb
import msgpack_numpy
import torch

sys.path.append("/home/4/ud02274/navigation/myss")
sys.path.insert(0, "/home/4/ud02274/navigation/myss/sound-spaces")

from ss_baselines.savi.config.default import get_config
from ss_baselines.savi.iprl_pretraining.offpolicy.iprl_pretraining_lmdb_dataset import IPRLPretrainingLMDBDataset
from videollama2.mm_utils import get_model_name_from_path
from videollama2.model.builder import load_pretrained_model
from videollama2.train import process_video
from videollama2.constants import MMODAL_TOKEN_INDEX
from videollama2.mm_utils import tokenizer_MMODAL_token


def get_visual_tensor(image_seq, processor, num_frames):
    tmp_image_seq = (image_seq.copy().squeeze(1)[:, :, :, :3] * 255).astype(np.uint8)

    ve_visual_tensor = process_video(
        tmp_image_seq.copy(),
        processor,
        "pad",
        len(tmp_image_seq),
    ).to(
        dtype=torch.float16,
        device='cuda',
        non_blocking=True,
    ) # (l, c, h, w)

    image_seq_len = len(tmp_image_seq)
    indices = np.arange(0, image_seq_len,  image_seq_len / num_frames).astype(int)
    fm_visual_tensor = process_video(
        tmp_image_seq[indices],
        processor,
        "pad",
        num_frames,
    ).to(
        dtype=torch.float16,
        device='cuda',
        non_blocking=True,
    ) # (l, c, h, w)
    return ve_visual_tensor, fm_visual_tensor


def main():
    # foundation_model_path = "DAMO-NLP-SG/VideoLLaMA2-7B-Base"
    # num_frames = 8
    foundation_model_path = "../xgenerator/data/models/video_llama2/finetune_videollama2_vllava_qlora-size128-e1-f16"
    num_frames = 16
    lmdb_dataset_path = "./data/lmdb_dataset/iprl_pretrain/val"
    save_path = "./data/lmdb_dataset/iprl_pretrain/testtesttest-bs1"
    split = "val_w_past_instruction"
    data_num = 500
    # split = "train_cr_0.2_w_past_instruction"
    # data_num = 100398
    batch_size = 3

    tokenizer, foundation_model, processor, _ = load_pretrained_model(
        foundation_model_path, None, get_model_name_from_path(foundation_model_path),
        device="cuda",
    )
    visual_encoder = foundation_model.get_model().get_vision_tower().vision_tower
    prompt = "[INST] <<SYS>>\n" \
        "A chat between a curious user and an artificial intelligence assistant." \
        "The assistant gives helpful, detailed, and polite answers to the user's questions." \
        "\n<</SYS>>\n\n <video>\nWhat is the camera wearer doing? [/INST]"
    input_ids = tokenizer_MMODAL_token(
        prompt, tokenizer, MMODAL_TOKEN_INDEX["VIDEO"], return_tensors='pt',
    ).unsqueeze(0).to(device="cuda")

    config = get_config("ss_baselines/savi/iprl_pretraining/config.yaml")
    dataset = IPRLPretrainingLMDBDataset(
        config=config.TASK_CONFIG,
        split=split,
        lmdb_dataset_path=lmdb_dataset_path,
        data_num=data_num,
        foundation_model_type=None,
        foundation_model_path=None,
        num_frames=None,
        environment_type="ss1-savi",
        device="cuda",
    )

    map_size = 8 * 1.1e12 # about 8 TB
    env = lmdb.open(save_path, map_size=int(map_size))

    max_ve_visual_tensor_len = 0
    ve_visual_tensors = []
    fm_visual_tensors = []
    save_idx = 0
    for i in trange(len(dataset)):
        x, _ = dataset[i]
        image_seq = x["image_seq"]
        ve_visual_tensor, fm_visual_tensor = get_visual_tensor(image_seq, processor, num_frames)

        if max_ve_visual_tensor_len < ve_visual_tensor.shape[0]:
            max_ve_visual_tensor_len = ve_visual_tensor.shape[0]
        ve_visual_tensors.append(ve_visual_tensor)
        fm_visual_tensors.append(fm_visual_tensor)

        if len(fm_visual_tensors) == batch_size or i == len(dataset)-1:
            if i == len(dataset)-1:
                batch_size = len(fm_visual_tensors)
            
            # Calculate visual features
            _, channel, height, width = ve_visual_tensor.shape
            ve_visual_tensor_lens = []
            input_visual_tensors = torch.zeros((batch_size, max_ve_visual_tensor_len, channel, height, width))
            for j, ve_visual_tensor in enumerate(ve_visual_tensors):
                ve_visual_tensor_lens.append(ve_visual_tensor.shape[0])
                input_visual_tensors[j, :ve_visual_tensor.shape[0], :, :, :] = ve_visual_tensor
            with torch.inference_mode():
                visual_features = visual_encoder(
                    input_visual_tensors.view(-1, channel, height, width).to("cuda"),
                    output_hidden_states=True,
                ).pooler_output # (l, 1024)
            visual_features = visual_features.view(batch_size, max_ve_visual_tensor_len, 1024)

            # Calculate logits
            outputs = foundation_model.generate(
                input_ids.repeat(batch_size, 1),
                images_or_videos=fm_visual_tensors,
                modal_list=['video' for _ in range(batch_size)],
                do_sample=True,
                temperature=0.2,
                # max_new_tokens=1024,
                max_new_tokens=40,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            logits = torch.stack(outputs.scores)
            
            max_ve_visual_tensor_len = 0
            ve_visual_tensors = []
            fm_visual_tensors = []

            assert logits.shape[1] == batch_size and visual_features.shape[0] == batch_size
            for j in range(batch_size):
                data = [
                    visual_features[j, :ve_visual_tensor_lens[j], :].to('cpu').detach().numpy().copy(),
                    logits[:, j, :].to('cpu').detach().numpy().copy(),
                ]

                with env.begin(write=True) as txn:
                    txn.put(
                        f"{save_idx}".encode(), 
                        msgpack_numpy.packb(
                            data, use_bin_type=True
                        ),
                    )
                save_idx += 1
            print(f"save_idx: {save_idx}!!")


if __name__=="__main__":
    main()