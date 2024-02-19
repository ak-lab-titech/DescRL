#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi"   # avnav, avwan, savi, random, ksaven
MODEL_NAME="ss1savi-savi"
DATASET_NAME="mp3d"
TEST_EPISODE_COUNT=1000
CKPT_NUM="100"

echo ENV=$ENV
echo MODEL_NAME=$MODEL_NAME
echo DATASET_NAME=$DATASET_NAME
echo TEST_EPISODE_COUNT=$TEST_EPISODE_COUNT
echo CKPT_NUM=$CKPT_NUM


if [ $ENV = "ss2-avnav" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_nav/run.py \
        --run-type eval \
        --exp-config ss_baselines/av_nav/config/audionav/replica/test_multi_goal/audiogoal_depth.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True \
        LOG_FILE data/models/ss2/replica/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/ss2/replica/$MODEL_NAME/tb_test \
        EVAL_CKPT_PATH_DIR data/models/ss2/replica/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "savi" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/audionav/savi.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        NUM_PROCESSES 1 \
        CONTINUOUS True \
        RL.DDPPO.pretrained False \
        LOG_FILE data/models/ss2/replica/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/ss2/replica/$MODEL_NAME/tb_test \
        EVAL_CKPT_PATH_DIR data/models/ss2/replica/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
        --model-dir data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME \
        EVAL_CKPT_PATH_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/test \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT test \
        RL.DDPPO.pretrained False \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        USE_SYNC_VECENV True \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven" ]; then
    python ss_baselines/saven/run.py \
        --run-type eval \
        --exp-config ss_baselines/saven/config/semantic_audionav/saven.yaml \
        --model-dir data/models/saven/$MODEL_NAME \
        EVAL_CKPT_PATH_DIR data/models/saven/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE data/models/saven/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/saven/$MODEL_NAME/test \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT test \
        RL.DDPPO.pretrained False \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        USE_SYNC_VECENV True
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_nav/run.py \
        --run-type eval \
        --exp-config ss_baselines/av_nav/config/semantic_audionav/mp3d/rgbd_ddppo.yaml \
        --model-dir data/models/ss1-savi/avnav/mp3d/$MODEL_NAME \
        TRAINER_NAME "AVNavTrainer" \
        EVAL_CKPT_PATH_DIR data/models/ss1-savi/avnav/mp3d/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE data/models/ss1-savi/avnav/mp3d/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/ss1-savi/avnav/mp3d/$MODEL_NAME/test \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT test \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        USE_SYNC_VECENV True
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "avnav" ]; then
    python ss_baselines/av_wan/run.py \
        --run-type eval \
        --exp-config ss_baselines/av_wan/config/semantic_audionav/mp3d/train_with_am.yaml \
        --model-dir data/models/ss1-savi/avwan/mp3d/$MODEL_NAME \
        EVAL_CKPT_PATH_DIR data/models/ss1-savi/avwan/mp3d/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        LOG_FILE data/models/ss1-savi/avwan/mp3d/$MODEL_NAME/test-ckpt$CKPT_NUM.log \
        TENSORBOARD_DIR data/models/ss1-savi/avwan/mp3d/$MODEL_NAME/test \
        NUM_PROCESSES 10 \
        CONTINUOUS False \
        EVAL.SPLIT test \
        TEST_EPISODE_COUNT $TEST_EPISODE_COUNT \
        USE_SYNC_VECENV True
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "random" ]; then
    python ss_baselines/common/simple_agents.py \
        --success-distance 1.0 \
        --task-config configs/audionav/av_nav/replica/pointgoal.yaml
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "random" ]; then
    python ss_baselines/common/simple_agents.py \
        --success-distance 1.0 \
        --task-config configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml \
        DATASET.SPLIT test
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi
