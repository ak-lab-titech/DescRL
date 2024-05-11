#!/bin/bash

ENV="ss1-savi" # ss1-avnav, ss2-avnav, ss1-savi, ss2-savi
MODEL="savi"   # avnav, avwan, savi, savi-on-iprl-pretraining, savi-off-iprl-pretraining
MODEL_NAME="ss1savi-savi"
DATASET_NAME="mp3d"
CKPT_NUM=154
VIDEO_NUM=5


pwd
cd ~/navigation/myss/sound-spaces

source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo ENV=$ENV
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo DATASET_NAME=$DATASET_NAME
echo CKPT_NUM=$CKPT_NUM
echo VIDEO_NUM=$VIDEO_NUM


if [ $ENV = "ss2-avnav" ] && [ $MODEL = "avnav" ]; then
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
        TEST_EPISODE_COUNT $VIDEO_NUM
elif [ $ENV = "ss2-avnav" ] && [ $MODEL = "savi" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/audionav/savi.yaml \
        --model-dir data/models/ss2/replica/$MODEL_NAME \
        CONTINUOUS True \
        NUM_PROCESSES 1 \
        RL.DDPPO.pretrained False \
        LOG_FILE data/models/ss2/replica/$MODEL_NAME/demo_video.log \
        TENSORBOARD_DIR data/models/ss2/replica/$MODEL_NAME/tb_demo_video \
        EVAL_CKPT_PATH_DIR data/models/ss2/replica/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False \
        TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE True \
        DISPLAY_RESOLUTION 128 \
        TEST_EPISODE_COUNT $VIDEO_NUM
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi" ]; then
    python ss_baselines/savi/run.py \
        --run-type eval \
        --exp-config ss_baselines/savi/config/semantic_audionav/savi.yaml \
        --model-dir data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME \
        CONTINUOUS False \
        NUM_PROCESSES 1 \
        EVAL.SPLIT test \
        RL.DDPPO.pretrained False \
        LOG_FILE data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/demo_video.log \
        TENSORBOARD_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/tb_demo_video_ckpt$CKPT_NUM \
        EVAL_CKPT_PATH_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False \
        TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE False \
        DISPLAY_RESOLUTION 512 \
        TEST_EPISODE_COUNT $VIDEO_NUM \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-on-iprl-pretraining" ]; then
    python ss_baselines/savi/iprl_pretraining/onpolicy/run.py \
        --run-type eval \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --model-dir data/models/ss1-savi/mp3d/$MODEL_NAME \
        CONTINUOUS False \
        NUM_PROCESSES 1 \
        EVAL.SPLIT test \
        RL.DDPPO.pretrained False \
        LOG_FILE data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/demo_video.log \
        TENSORBOARD_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/tb_demo_video_ckpt$CKPT_NUM \
        EVAL_CKPT_PATH_DIR data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS False \
        TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE False \
        DISPLAY_RESOLUTION 512 \
        TEST_EPISODE_COUNT $VIDEO_NUM \
        RL.PPO.use_belief_predictor True \
        RL.PPO.INSTRUCTION_PREDICTOR.iprl_use_gt_D False \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "savi-off-iprl-pretraining" ]; then
    # 注意：beliefを用いず、GTのcateogryとlocationを入力してtestしている
    python ss_baselines/savi/iprl_pretraining/offpolicy/off_policy_eval.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --model-dir  data/models/ss1-savi/mp3d/$MODEL_NAME \
        --indices 1 2 3 \
        --past-or-future future \
        --dataset-type test \
        CONTINUOUS False \
        NUM_PROCESSES 1 \
        RL.DDPPO.pretrained False \
        RL.DDPPO.pretrained_weights data/models/ss1-savi/$DATASET_NAME/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        DISPLAY_RESOLUTION 512 \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
elif [ $ENV = "ss1-savi" ] && [ $MODEL = "ksaven-off-iprl-pretraining" ]; then
    python ss_baselines/saven/iprl_pretraining/offpolicy/off_policy_eval.py \
        --config ss_baselines/saven/iprl_pretraining/config.yaml \
        --model-dir  data/models/saven/$MODEL_NAME \
        --indices 0 1 2 3 4 5 6 7 8 9 \
        --past-or-future past \
        --dataset-type test \
        CONTINUOUS False \
        NUM_PROCESSES 1 \
        RL.DDPPO.pretrained False \
        RL.DDPPO.pretrained_weights data/models/saven/$MODEL_NAME/data/ckpt.$CKPT_NUM.pth \
        DISPLAY_RESOLUTION 512 \
        RL.PPO.INSTRUCTION_PREDICTOR.feedback "student"
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL.
fi
