#!/usr/bin/env bash

#PBS -lwalltime=1:00:00
#PBS -lselect=1:ncpus=32:mem=62gb

#PBS -o logs/
#PBS -e logs/

# https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/computing/job-sizing-guidance/high-throughput/

set -ex
hostname
printenv
date
lscpu

export num_threads=32

ROOT_DIR=$PBS_O_WORKDIR
SIF_PATH=$ROOT_DIR/singularity/image.sif

workdir=$ROOT_DIR
cd $workdir

singularity exec --containall \
    -B $ROOT_DIR:$ROOT_DIR \
    -B $TMPDIR:/tmp \
    $SIF_PATH \
    bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate synthia-paper && cd $workdir && \
    python -u src/plot.py"
