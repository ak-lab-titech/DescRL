#!/bin/bash


pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

# AV-Nav
python ss_baselines/av_nav/run.py \
    --exp-config ss_baselines/av_nav/config/audionav/replica/train_multi_goal/audiogoal_depth_ddppo.yaml \
    --model-dir data/models/ss2/replica/2g-avnav-wSDM \
    CONTINUOUS True \


# AV-WaN
# python ss_baselines/av_wan/run.py \
#     --exp-config ss_baselines/av_wan/config/audionav/replica/train_with_am.yaml \
#     --model-dir data/models/ss2/replica/test-savi1 \
#     CONTINUOUS True \


# SAVi first stage
# python ss_baselines/savi/run.py \
#     --exp-config ss_baselines/savi/config/audionav/savi_pretraining.yaml \
#     --model-dir data/models/ss2/replica/ \
#     CONTINUOUS True \

# SAVi second stage
# python ss_baselines/savi/run.py \
#     --exp-config ss_baselines/savi/config/audionav/savi.yaml \
#     --model-dir data/models/ss2/replica/ \
#     CONTINUOUS True \
