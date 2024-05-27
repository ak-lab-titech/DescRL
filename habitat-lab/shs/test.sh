#!/bin/bash

ENV="habitat-objnav"
SCENE_DATASET="mp3d"
MODEL="rnn"
MODEL_NAME="baseline"
CKPT_NUM=50

echo ENV=$ENV
echo MODEL=$MODEL
echo SCENE_DATASET=$SCENE_DATASET
echo MODEL_NAME=$MODEL_NAME
echo CKPT_NUM=$CKPT_NUM


source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


if [ $ENV = "habitat-objnav" ] && [ $MODEL = "rnn" ] && [ $SCENE_DATASET = "mp3d" ]; then
    cd ~/navigation/myss/habitat-lab
    python habitat_baselines/run.py \
        --run-type eval \
        --exp-config ./habitat_baselines/config/objectnav/ddppo_objectnav.yaml \
        --model-dir ./data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME \
        EVAL_CKPT_PATH_DIR data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE ./data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR ./data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/tb-test-ckpt$CKPT_NUM \
        EVAL.SPLIT val_mini \
        TRAINER_NAME ppo \
        NUM_ENVIRONMENTS 1 \
        NUM_PROCESSES 1
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi




