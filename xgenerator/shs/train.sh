#!/bin/bash


MODEL="video_llava"
MODEL_NAME="qlora-test"
TRAIN_NP=4

export NP=$TRAIN_NP
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo NP=$NP

cd ~/navigation/myss/xgenerator

if [ $MODEL = "transformer_speaker" ] || [ $MODEL = "speaker" ]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nnodes=1 \
        --nproc_per_node=$NP \
        $MODEL/train.py \
        --config-path $MODEL/config.yaml \
        --model-name $MODEL_NAME
elif [ $MODEL = "video_llava" ] || [ $MODEL = "video_blip" ] ; then
    python $MODEL/qlora.py \
        --model-name $MODEL_NAME
else
    echo ERROR: $MODEL
fi