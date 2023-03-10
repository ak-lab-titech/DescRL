#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


python ./scripts/plot_tb_data.py \
    --yaml_path ./configs/audionav/av_nav/replica/plot_tb_data.yaml
