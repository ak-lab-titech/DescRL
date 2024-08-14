#!/bin/bash

CMD="vlnce_test_dataset"

# vlnce_test_dataset
INSTRUCTION_PREDICTOR_TYPE="video-llama2"
ENVIRONMENT_TYPE="vlnce"
DATASET_NAME="val_unseen"

# # generate_instruction
# ENVIRONMENT_TYPE="habitat-objnav"
# DATASET_NAME="train_w_past_instruction_split1_hogehoge"
# FUTURE_OR_PAST="past"

# # offpolicy_episode_dataset
# ENVIRONMENT_TYPE="habitat-objnav"
# DATASET_NAME="train"
# START_INDEX=0


pwd
cd ~/navigation/myss/sound-spaces

source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo CMD=$CMD
echo DATASET_NAME=$DATASET_NAME
echo START_INDEX=$START_INDEX
echo FUTURE_OR_PAST=$FUTURE_OR_PAST
echo INSTRUCTION_PREDICTOR_TYPE=$INSTRUCTION_PREDICTOR_TYPE
echo ENVIRONMENT_TYPE=$ENVIRONMENT_TYPE
echo DATASET_NAME=$DATASET_NAME


if [ $CMD = "replica_dataset" ]; then
    python soundspaces/datasets/make_dataset.py \
        --yaml_path ./configs/audionav/av_nav/replica/make_dataset.yaml
elif [ $CMD = "generate_instruction" ] && [ $ENVIRONMENT_TYPE = "ss1-savi" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_instructions.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --save-dataset-path ./data/datasets/semantic_audionav/mp3d/v1/$DATASET_NAME \
        --future-or-past $FUTURE_OR_PAST
elif [ $CMD = "offpolicy_episode_dataset" ] && [ $ENVIRONMENT_TYPE = "ss1-savi" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_episode_dataset.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --save-dataset-path ./data/lmdb_dataset/iprl_pretrain/$DATASET_NAME \
        --start-index $START_INDEX
elif [ $CMD = "KD_FM_dataset" ] && [ $ENVIRONMENT_TYPE == "ss1-savi" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_fm_KD_dataset.py
elif [ $CMD = "generate_instruction" ] && [ $ENVIRONMENT_TYPE = "habitat-objnav" ]; then
    cd ~/navigation/myss
    python sound-spaces/ss_baselines/savi/iprl_pretraining/scripts/make_instructions.py \
        --config habitat-lab/habitat_baselines/config/objectnav_iprl_pretraining/config.yaml \
        --save-dataset-path ./habitat-lab/data/datasets/objectnav/mp3d/v1/$DATASET_NAME \
        --future-or-past past
elif [ $CMD = "offpolicy_episode_dataset" ] && [ $ENVIRONMENT_TYPE = "habitat-objnav" ]; then
    cd ~/navigation/myss
    python sound-spaces/ss_baselines/savi/iprl_pretraining/scripts/make_episode_dataset.py \
        --config habitat-lab/habitat_baselines/config/objectnav_iprl_pretraining/config.yaml \
        --save-dataset-path habitat-lab/data/lmdb_dataset/iprl_pretrain/$DATASET_NAME \
        --start-index $START_INDEX
elif [ $CMD = "sub_episode_dataset" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_sub_episode_dataset.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --source-dataset-path ./data/lmdb_dataset/iprl_pretrain/train \
        --source-data-num 502103 \
        --data-num 10000
elif [ $CMD = "vlnce_test_dataset" ]; then
    cd ~/navigation/myss
    python ./sound-spaces/scripts/make_vlnce_test_dataset.py \
        --config ./sound-spaces/scripts/configs/make_vlnce_test_dataset.yaml \
        --save-dataset-path ../my-VLN-CE/data/datasets/$CMD/$INSTRUCTION_PREDICTOR_TYPE/$ENVIRONMENT_TYPE/$DATASET_NAME \
        --instruction-predictor-type $INSTRUCTION_PREDICTOR_TYPE \
        --environment-type $ENVIRONMENT_TYPE
else
    echo ERROR: CMD=$CMD, ENVIRONMENT_TYPE=$ENVIRONMENT_TYPE
fi
