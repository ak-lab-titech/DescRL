#!/bin/bash


pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


python soundspaces/datasets/make_dataset.py \
    --yaml_path ./configs/audionav/av_nav/replica/make_dataset.yaml


# python debug.py
