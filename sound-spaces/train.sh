#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

python ss_baselines/av_nav/run.py \
    --exp-config ss_baselines/av_nav/config/audionav/replica/train_multi_goal/audiogoal_depth_ddppo.yaml \
    --model-dir data/models/ss2/replica/2g-tsubame-actC1-NP4-DirectMap-coef1000 \
    CONTINUOUS True
