#!/bin/bash

CMD="offpolicy_episode_dataset"
DATASET_NAME="train_dataset"


pwd
cd ~/av-nav/myss/sound-spaces

source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo CMD=$CMD
echo DATASET_NAME=$DATASET_NAME


if [ $CMD = "replica_dataset" ]; then
    python soundspaces/datasets/make_dataset.py \
        --yaml_path ./configs/audionav/av_nav/replica/make_dataset.yaml
elif [ $CMD = "generate_instruction" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_instructions.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --save-dataset-path ./data/datasets/semantic_audionav/mp3d/v1/$DATASET_NAME
elif [ $CMD = "offpolicy_episode_dataset" ]; then
    python ss_baselines/savi/iprl_pretraining/scripts/make_episode_dataset.py \
        --config ss_baselines/savi/iprl_pretraining/config.yaml \
        --save-dataset-path ./data/lmdb_dataset/$DATASET_NAME
else
    echo ERROR: CMD=$CMD
fi
