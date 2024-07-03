#!/bin/bash

ENV="habitat-objnav"
SCENE_DATASET="mp3d"
MODEL="rnn"
MODEL_NAME="baseline"
CKPT_NUM=50
TEST_EPISODE_COUNT=1000

echo ENV=$ENV
echo MODEL=$MODEL
echo SCENE_DATASET=$SCENE_DATASET
echo MODEL_NAME=$MODEL_NAME
echo CKPT_NUM=$CKPT_NUM
echo TEST_EPISODE_COUNT=$TEST_EPISODE_COUNT


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
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        EVAL.SPLIT my_test \
        TRAINER_NAME ppo \
        NUM_ENVIRONMENTS 1 \
        NUM_PROCESSES 1
elif [ $ENV = "habitat-objnav" ] && [ $MODEL = "smt" ] && [ $SCENE_DATASET = "mp3d" ]; then
    cd ~/navigation/myss
    python ./sound-spaces/ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ./habitat-lab/habitat_baselines/config/objectnav/ddppo_smt.yaml \
        --model-dir ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME \
        EVAL_CKPT_PATH_DIR ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/test.log \
        TENSORBOARD_DIR ./habitat-lab/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/tb_test \
        EVAL.SPLIT my_test \
        NUM_ENVIRONMENTS 7 \
        NUM_PROCESSES 7 \
        DISPLAY_RESOLUTION 640 \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi
