#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi-1st"   # avnav, avwan, savi-1st, savi-2nd, ksaven-1st, ksaven-2nd
MODEL_NAME="ss1savi-savi"
DATASET_NAME="mp3d"
PREV_CKPT_IND=-1 # savi-2ndなら途中からになるように変更するべき
TEST_EPISODE_COUNT=100

echo ENV=$ENV
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo DATASET_NAME=$DATASET_NAME
echo PREV_CKPT_IND=$PREV_CKPT_IND


if [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-1st" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi_pretraining.yaml \
        --model-dir data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        LOG_FILE data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT val \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-2nd" ]; then
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
        RL.DDPPO.pretrained False \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-1st" ]; then
    python ss_baselines/saven/run.py \
        --run-type eval \
        --exp-config ss_baselines/saven/config/semantic_audionav/saven_pretraining.yaml \
        --model-dir data/models/saven/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        EVAL.SPLIT val \
        LOG_FILE data/models/saven/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/saven/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-2nd" ]; then
    python ss_baselines/saven/run.py \
        --run-type eval \
        --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml \
        --model-dir data/models/saven/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        EVAL.SPLIT val \
        LOG_FILE data/models/saven/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/saven/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        RL.DDPPO.pretrained False
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_nav/run.py \
        --run-type eval \
        --exp-config ss_baselines/av_nav/config/semantic_audionav/mp3d/rgbd_ddppo.yaml \
        --model-dir data/models/ss1-savi/avnav/mp3d/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        TRAINER_NAME "AVNavTrainer" \
        LOG_FILE data/models/ss1-savi/avnav/mp3d/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/ss1-savi/avnav/mp3d/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT val \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avwan" ]; then
    python ss_baselines/av_wan/run.py \
        --run-type eval \
        --exp-config ss_baselines/av_wan/config/semantic_audionav/mp3d/train_with_am.yaml \
        --model-dir data/models/ss1-savi/avwan/mp3d/$MODEL_NAME \
        --prev-ckpt-ind $PREV_CKPT_IND \
        LOG_FILE data/models/ss1-savi/avwan/mp3d/$MODEL_NAME/eval.log \
        TENSORBOARD_DIR data/models/ss1-savi/avwan/mp3d/$MODEL_NAME/tb_eval \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT val \
        USE_SYNC_VECENV True \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi

