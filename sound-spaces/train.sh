#!/bin/bash

ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi-2nd"
MODEL_NAME="ss1savi-savi-iprl"
# PRETRAINED_MODEL_NAME="ss1savi-savi-1st-v2"
# CKPT_NUM="115"
PRETRAINED_MODEL_NAME="ss1savi-savi-1st-duraion"
CKPT_NUM="55"
# PRETRAINED_MODEL_NAME="ss1savi-ksaven-1st-duration"
# CKPT_NUM="54"

# for off-policy iprl pre-training
LOG_INTERVAL=1
SAVE_INTERVAL=100
VAL_INTERVAL=100

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo ENV=$ENV
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo PRETRAINED_MODEL_NAME=$PRETRAINED_MODEL_NAME
echo CKPT_NUM=$CKPT_NUM
echo LOG_INTERVAL=$LOG_INTERVAL
echo SAVE_INTERVAL=$SAVE_INTERVAL
echo VAL_INTERVAL=$VAL_INTERVAL


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
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_nav/run.py \
        --exp-config ss_baselines/av_nav/config/semantic_audionav/mp3d/rgbd_ddppo.yaml \
        --model-dir data/models/ss1-savi/avnav/mp3d/$MODEL_NAME
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avwan" ]; then
    python ss_baselines/av_wan/run.py \
        --exp-config ss_baselines/av_wan/config/semantic_audionav/mp3d/train_with_am.yaml \
        --model-dir data/models/ss1-savi/avwan/mp3d/$MODEL_NAME 
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-1st" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-2nd" ]; then
    python ss_baselines/savi/run.py \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        RL.DDPPO.pretrained_weights "data/models/ss1-savi/mp3d/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-iprl-pre-onpolicy" ]; then
    python ss_baselines/savi/iprl_pretraining/onpolicy/run.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        RL.DDPPO.pretrained_weights "data/models/ss1-savi/mp3d/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-iprl-pre-offpolicy" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nnodes=1 \
        --nproc_per_node=$NP \
       ss_baselines/savi/iprl_pretraining/offpolicy/off_policy_train.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        --use-lmdb True \
        --log-interval $LOG_INTERVAL \
        --save-interval $SAVE_INTERVAL \
        --val-interval $VAL_INTERVAL \
        RL.DDPPO.pretrained_weights "data/models/ss1-savi/mp3d/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-vision" ]; then
    python ss_baselines/saven/pretraining/vision_model_trainer.py \
        --run-type train \
        --model-dir data/models/saven/$MODEL_NAME \
        --use-multiple-GPU
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-audio" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nnodes=1 \
        --nproc_per_node=$NP \
        ss_baselines/saven/pretraining/audio_model_trainer.py \
            --run-type train \
            --model-dir data/models/saven/$MODEL_NAME \
            --use-multiple-GPU
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-1st" ]; then
    python ss_baselines/saven/run.py \
        --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml \
        --model-dir data/models/saven/$MODEL_NAME
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-2nd" ]; then
    python ss_baselines/saven/run.py \
        --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml \
        --model-dir data/models/saven/$MODEL_NAME \
        RL.DDPPO.pretrained_weights "data/models/saven/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-iprl-pre-offpolicy" ]; then
    python ss_baselines/saven/iprl_pretraining/offpolicy/off_policy_train.py \
        --config ss_baselines/saven/iprl_pretraining/config.yaml \
        --model-dir data/models/saven/$MODEL_NAME \
        --use-lmdb True \
        --log-interval $LOG_INTERVAL \
        --save-interval $SAVE_INTERVAL \
        --val-interval $VAL_INTERVAL \
        RL.DDPPO.pretrained_weights "data/models/saven/$PRETRAINED_MODEL_NAME/data/ckpt.$CKPT_NUM.pth"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi
