#!/bin/bash


MODEL="transformer_speaker"
MODEL_NAME="tf-speaker"
EVAL_NUM=3
EVAL_CKPT_NUM=70
EVAL_SEED=0

echo MODEL_NAME=$MODEL_NAME
echo EVAL_NUM=$EVAL_NUM
echo CKPT_NUM=$EVAL_CKPT_NUM
echo EVAL_SEED=$EVAL_SEED

cd ~/navigation/myss/xgenerator

if [ $MODEL = "transformer_speaker" ] || [ $MODEL = "speaker" ]; then
    python $MODEL/eval.py \
        --eval-num $EVAL_NUM \
        --ckpt-num $EVAL_CKPT_NUM \
        --model-path ./data/models/$MODEL_NAME \
        --random-seed $EVAL_SEED
else
    echo ERROR: $MODEL
fi