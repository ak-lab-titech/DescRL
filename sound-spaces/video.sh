#!/bin/bash

pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

MODEL_NAME="2g-v2-DMGT-iros"
CKPT_NUM="25"


# AV-Nav
python ss_baselines/av_nav/run.py \
    --run-type eval \
    --exp-config ss_baselines/av_nav/config/audionav/replica/test_multi_goal/audiogoal_depth.yaml \
    --model-dir data/models/ss2/replica/$MODEL_NAME \
    CONTINUOUS True \
    LOG_FILE data/models/ss2/replica/$MODEL_NAME/demo_video.log \
    TENSORBOARD_DIR data/models/ss2/replica/$MODEL_NAME/tb_demo_video \
    EVAL_CKPT_PATH_DIR data/models/ss2/replica/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
    TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False \
    TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE True \
    DISPLAY_RESOLUTION 512 \
    TEST_EPISODE_COUNT 10


# SAVi
# python ss_baselines/savi/run.py \
#     --run-type eval \
#     --exp-config ss_baselines/savi/config/audionav/savi.yaml \
#     --model-dir data/models/ss2/replica/$MODEL_NAME \
#     CONTINUOUS True \
#     NUM_PROCESSES 1 \
#     RL.DDPPO.pretrained False \
#     LOG_FILE data/models/ss2/replica/$MODEL_NAME/demo_video.log \
#     TENSORBOARD_DIR data/models/ss2/replica/$MODEL_NAME/tb_demo_video \
#     EVAL_CKPT_PATH_DIR data/models/ss2/replica/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
#     TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False \
#     TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE True \
#     DISPLAY_RESOLUTION 128 \
#     TEST_EPISODE_COUNT 5
