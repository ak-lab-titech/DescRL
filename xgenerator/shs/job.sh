#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -j y
#$ -l h_rt=10:00:00
#$ -o output/o.$JOB_ID


module load cuda
module load nccl
module load cudnn
module load openmpi/5.0.2-intel


CMD="train"


cd ~/navigation/myss/xgenerator/shs
pwd
source /gs/fs/tga-aklab/hkondo/anaconda3/etc/profile.d/conda.sh
conda activate av-nav
echo CMD=$CMD

if [ $CMD = "train" ]; then
    ./train.sh
elif [ $CMD = "eval" ]; then
    ./eval.sh
else
    echo ERROR: CMD must be 'train' or 'eval', not $CMD.
fi