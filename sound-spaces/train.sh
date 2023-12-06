#!/bin/bash

ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi-2nd"   # avnav, avwan, savi-1st, savi-2nd, savi-iprl-pretraining
MODEL_NAME="ss1savi-savi-iprl"
PRETRAINED_MODEL_NAME="ss1savi-savi-1st"
CKPT_NUM="106"

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo ENV=$ENV
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo PRETRAINED_MODEL_NAME=$PRETRAINED_MODEL_NAME
echo CKPT_NUM=$CKPT_NUM


if [ $ENV = "ss2-avnav" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_nav/run.py \
        --exp-config ss_baselines/av_nav/config/audionav/replica/train_multi_goal/audiogoal_depth_ddppo.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "avwan" ]; then
    python ss_baselines/av_wan/run.py \
        --exp-config ss_baselines/av_wan/config/audionav/replica/train_with_am.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "savi-1st" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/audionav/savi_pretraining.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "savi-2nd" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/audionav/savi.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-1st" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-2nd" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        RL.DDPPO.pretrained_weights "data/models/ss1-savi/mp3d/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-iprl-pretraining" ]; then
    python ss_baselines/savi/iprl_pretraining/run.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        RL.DDPPO.pretrained_weights "data/models/ss1-savi/mp3d/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi
