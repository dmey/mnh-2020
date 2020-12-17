#!/usr/bin/env bash

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -J 1-2

#PBS -o logs/
#PBS -e logs/

# https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/computing/job-sizing-guidance/high-throughput/

if [[ "$DRY_RUN" == "" ]]; then

set -ex
hostname
printenv
date
lscpu

export num_threads=32

ROOT_DIR=$PBS_O_WORKDIR
SIF_PATH=$ROOT_DIR/singularity/image.sif

workdir=$ROOT_DIR/notebooks
cd $workdir
output_dir=../results/notebooks
mkdir -p $output_dir

fi

i=0

for data_fraction in 1.0; do
    ((i=i+1))
    if [[ "$DRY_RUN" == "1" ]]; then
        continue
    fi
    if [[ $i -eq $PBS_ARRAY_INDEX ]]; then
        export SINGULARITYENV_data_fraction=$data_fraction
        
        singularity exec --containall \
            -B $ROOT_DIR:$ROOT_DIR \
            -B $TMPDIR:/tmp \
            $SIF_PATH \
            bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate synthia-paper && cd $workdir && \
            PYTHONHASHSEED=0 jupyter nbconvert --to html --execute ml_control.ipynb \
            --output=$output_dir/ml-control-data_fraction=$data_fraction \
            --ExecutePreprocessor.timeout=259200 --ExecutePreprocessor.iopub_timeout=300"

        exit 0
    fi
done
if [[ "$DRY_RUN" == "1" ]]; then
    echo "#PBS -J 1-$i"
fi
