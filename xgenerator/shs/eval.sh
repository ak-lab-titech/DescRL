#!/bin/bash


MODEL="transformer_speaker"
MODEL_NAME="tf-speaker"
EVAL_NUM=3
EVAL_CKPT_NUM=70
EVAL_SEED=0
$BEMA_NUM=1
$TOP_K=1
$TOP_P=
$TEMPERATURE=1

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
        --random-seed $EVAL_SEED \
        --beam-num $BEAM_NUM \
        --top-k $TOP_K \
        --top-p $TOP_P \
        --temperature $TEMPERATURE 
else
    echo ERROR: $MODEL
fi
