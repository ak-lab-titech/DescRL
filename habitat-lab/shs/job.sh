#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=24:00:00
#$ -o output/o.$JOB_ID


CMD="multi-gpu-train"


module load cuda
module load nccl
module load cudnn
module load openmpi/5.0.2-intel

export NNODES=$NHOSTS
export NPERNODE=4
export NP=$(($NPERNODE * $NNODES))
export MASTER_ADDR=`head -n 1 $SGE_JOB_SPOOL_DIR/pe_hostfile | cut -d " " -f 1`
export MASTER_PORT=$((10000+ ($JOB_ID % 50000)))
echo NNODES=$NNODES
echo NPERNODE=$NPERNODE
echo NP=$NP
echo MASTERADDR=$MASTER_ADDR
echo MASTERPORT=$MASTER_PORT
echo CMD=$CMD

cd ~/navigation/myss/habitat-lab/shs

if [ $CMD = "multi-gpu-train" ]; then
    mpirun -np $NP -npernode $NPERNODE \
        ./train.sh
elif [ $CMD = "single-gpu-train" ]; then
    ./train.sh
elif [ $CMD = "eval" ]; then
    ./eval.sh
elif [ $CMD = "test" ]; then
    ./test.sh
elif [ $CMD = "video" ]; then
    ./video.sh
else
    echo ERROR: CMD=$CMD.
fi
