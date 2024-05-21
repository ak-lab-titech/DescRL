#!/bin/bash

ENV="ss1-objnav"
SCENE_DATASET="mp3d"
MODEL="smt"
MODEL_NAME="baseline-ss1-FEPRL"
PREV_CKPT_IND=-1
TEST_EPISODE_COUNT=100

echo ENV=$ENV
echo MODEL=$MODEL
echo SCENE_DATASET=$SCENE_DATASET
echo MODEL_NAME=$MODEL_NAME


source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

if [ $ENV = "ss1-objnav" ] && [ $MODEL = "smt" ]; then
    cd ~/navigation/myss
    python ./sound-spaces/ss_baselines/savi/run.py \
        --exp-config ./habitat-lab/habitat_baselines/config/objectnav/ddppo_smt.yaml \
        --model-dir ./sound-spaces/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME \
        --run-type eval \
        --prev-ckpt-ind $PREV_CKPT_IND \
        LOG_FILE ./sound-spaces/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR ./sound-spaces/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME/tb_eval \
        EVAL.SPLIT val \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi


