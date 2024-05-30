#!/bin/bash

ENV="habitat-objnav"
SCENE_DATASET="mp3d"
MODEL="smt"
MODEL_NAME="baseline"
CKPT_NUM=50
VIDEO_NUM=5

echo ENV=$ENV
echo MODEL=$MODEL
echo SCENE_DATASET=$SCENE_DATASET
echo MODEL_NAME=$MODEL_NAME
echo CKPT_NUM=$CKPT_NUM
echo VIDEO_NUM=$VIDEO_NUM

source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


if [ $ENV = "habitat-objnav" ] && [ $MODEL = "smt" ] && [ $SCENE_DATASET = "mp3d" ]; then
    cd ~/navigation/myss
    python ./sound-spaces/ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ./habitat-lab/habitat_baselines/config/objectnav/ddppo_smt.yaml \
        --model-dir ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME \
        NUM_ENVIRONMENTS 1 \
        NUM_PROCESSES 1 \
        EVAL.SPLIT val_mini \
        LOG_FILE ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/video_ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/tb_video_ckpt$CKPT_NUM \
        EVAL_CKPT_PATH_DIR ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        RL.DDPPO.pretrained False \
        DISPLAY_RESOLUTION 640 \
        TEST_EPISODE_COUNT $VIDEO_NUM \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL, SCENE_DATASET=$SCENE_DATASET.
fi

