#!/bin/sh
#$ -cwd
#$ -l q_node=1
#$ -j y
#$ -l h_rt=06:00:00
#$ -o output/o.$JOB_ID


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


singularity exec --nv \
        --bind /gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/:/gs/hs0/tga-aklab/datasets/mp3d/v1/tasks/ \
        --bind /gs/hs0/tga-aklab/hkondo/av-nav/:/gs/hs0/tga-aklab/hkondo/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/vlnce/:/gs/hs0/tga-aklab/hkondo/vlnce/ \
        --bind /gs/hs0/tga-aklab/hkondo/xgenerator/:/gs/hs0/tga-aklab/hkondo/xgenerator/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/av-nav/:/gs/hs0/tga-aklab/hkondo/anaconda/av-nav/ \
        --bind /gs/hs0/tga-aklab/hkondo/anaconda/pkgs:/gs/hs0/tga-aklab/hkondo/anaconda/pkgs \
        /gs/hs0/tga-aklab/hkondo/nvidia_cudagl.img \
        ./train.sh