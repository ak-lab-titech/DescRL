#!/bin/sh
#$ -cwd
#$ -l node_f=1
#$ -j y
#$ -l h_rt=00:30:00
#$ -o output/o.$JOB_ID


CMD="multi-gpu-train"


module load cuda
module load nccl
module load cudnn
module load openmpi/5.0.2-intel


export NCCL_TIMEOUT_MS=1800000
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
elif [ $CMD = "make-dataset" ]; then
    ./make_dataset.sh
elif [ $CMD = "plot-tb-data" ]; then
    ./plot_tb_data.sh
else
    echo ERROR: CMD must be 'multi-gpu-train', 'single-gpu-train', 'test', 'video', 'make_dataset', or 'plot_tb_data', not $CMD.
fi
