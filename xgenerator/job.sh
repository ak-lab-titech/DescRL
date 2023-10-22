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


MODEL_NAME="speaker"
export NP=8


cd ~/av-nav/myss/xgenerator
pwd
source ~/anaconda3/etc/profile.d/conda.sh
conda activate av-nav

echo MODEL_NAME=$MODEL_NAME
echo NP=$NP

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
        --nnodes=1 \
        --nproc_per_node=$NP \
        speaker/train.py \
        --config-path \
        ./speaker/config.yaml \
        --model-name $MODEL_NAME