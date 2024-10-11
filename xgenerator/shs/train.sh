#!/bin/bash


MODEL="video_llama2"
MODEL_NAME="finetune_videollama2_vllava_qlora-test" # video_llama2の時はfinetune_videollama2_vllava_qloraを入れないとかも？
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
elif [ $MODEL = "video_llama2" ] || [ $MODEL = "video_llava" ] || [ $MODEL = "video_blip" ] ; then
    python $MODEL/qlora.py \
        --model-name $MODEL_NAME
else
    echo ERROR: $MODEL
fi
