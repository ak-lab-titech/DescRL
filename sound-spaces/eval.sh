#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi"   # avnav, avwan, savi
MODEL_NAME="ss1savi-savi"
DATASET_NAME="mp3d"
PREV_CKPT_IND=-1
TEST_EPISODE_COUNT=100

echo ENV=$ENV
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo DATASET_NAME=$DATASET_NAME
echo PREV_CKPT_IND=$PREV_CKPT_IND


if [ $ENV = "ss1-savi" && $MODEL = "savi" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
        --model-dir data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        LOG_FILE data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT val \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi

