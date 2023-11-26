#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o output/o.$JOB_ID


module load cuda
module load gcc/8.3.0-cuda
module load singularity
module load nccl
module load cudnn
module load openmpi/3.1.4-opa10.10


CMD="train"
MODEL="transformer_speaker" # speaker, transformer_speaker

MODEL_NAME="tf-speaker"
TRAIN_NP=4
EVAL_NUM=3
EVAL_CKPT_NUM=70
EVAL_SEED=0


cd ~/av-nav/myss/xgenerator
pwd
source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav
echo CMD=$CMD
echo MODEL=$MODEL

if [ $CMD = "train" ]; then
        export NP=$TRAIN_NP
        echo MODEL_NAME=$MODEL_NAME
        echo NP=$NP
        CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
                --nnodes=1 \
                --nproc_per_node=$NP \
                $MODEL/train.py \
                --config-path $MODEL/config.yaml \
                --model-name $MODEL_NAME
elif [ $CMD = "eval" ]; then
        echo MODEL_NAME=$MODEL_NAME
        echo EVAL_NUM=$EVAL_NUM
        echo CKPT_NUM=$EVAL_CKPT_NUM
        echo EVAL_SEED=$EVAL_SEED
        python $MODEL/eval.py \
            --eval-num $EVAL_NUM \
            --ckpt-num $EVAL_CKPT_NUM \
            --model-path ./data/models/$MODEL_NAME \
            --random-seed $EVAL_SEED
else
    echo ERROR: CMD must be 'train' or 'eval', not $CMD.
fi