#!/bin/sh
#$ -cwd
#$ -l f_node=1
#$ -j y
#$ -l h_rt=00:30:00
#$ -o output/o.$JOB_ID


CMD="multi-gpu-train" 


module load cuda
module load gcc/8.3.0-cuda
module load singularity
module load nccl
module load cudnn
module load openmpi/3.1.4-opa10.10

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

if [ $CMD = "multi-gpu-train" ]; then
    mpirun -np $NP -npernode $NPERNODE \
        singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./train.sh
elif [ $CMD = "single-gpu-train" ]; then
    singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./train.sh
elif [ $CMD = "test" ]; then
    singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        ~/nvidia_cudagl.img \
        ./test.sh
elif [ $CMD = "video" ]; then
    singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./video.sh
elif [ $CMD = "make_dataset" ]; then
    singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./make_dataset.sh
elif [ $CMD = "plot_tb_data" ]; then
    singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./plot_tb_data.sh
else
    echo ERROR: CMD must be 'multi-gpu-train', 'single-gpu-train', 'test', 'video', 'make_dataset', or 'plot_tb_data', not $CMD.
fi
