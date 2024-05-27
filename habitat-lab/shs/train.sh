#!/bin/bash

ENV="habitat-objnav"
SCENE_DATASET="mp3d"
MODEL="rnn"
MODEL_NAME="baseline"

echo ENV=$ENV
echo MODEL=$MODEL
echo SCENE_DATASET=$SCENE_DATASET
echo MODEL_NAME=$MODEL_NAME


source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav


if [ $ENV = "ss1-objnav" ] && [ $MODEL = "smt" ] && [ $SCENE_DATASET = "mp3d" ]; then
    cd ~/navigation/myss
    python ./sound-spaces/ss_baselines/savi/run.py \
        --exp-config ./habitat-lab/habitat_baselines/config/objectnav/ddppo_smt.yaml \
        --model-dir ./sound-spaces/data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME
elif [ $ENV = "habitat-objnav" ] && [ $MODEL = "rnn" ] && [ $SCENE_DATASET = "mp3d" ]; then
    cd ~/navigation/myss/habitat-lab
    python habitat_baselines/run.py \
        --run-type train \
        --exp-config ./habitat_baselines/config/objectnav/ddppo_objectnav.yaml \
        --model-dir ./data/models/$ENV/$SCENE_DATASET/$MODEL/$MODEL_NAME
else
    echo ERROR: ENV=$ENV, MODEL=$MODEL, SCENE_DATASET=$SCENE_DATASET.
fi
