#!/bin/bash

pwd
cd ~/navigation/myss/sound-spaces

source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


python ./scripts/plot_tb_data.py \
    --yaml_path ./configs/audionav/av_nav/replica/plot_tb_data.yaml
