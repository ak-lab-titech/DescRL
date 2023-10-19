#!/bin/bash

MODEL_NAME="speaker"

pwd
cd ~/av-nav/myss/xgenerator

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

python speaker/train.py \
    --config-path ./speaker/config.yaml \
    --model-name $MODEL_NAME
