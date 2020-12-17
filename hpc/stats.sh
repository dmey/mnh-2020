#!/usr/bin/env bash

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -J 1-600

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

for copula_type in parametric tll gaussian; do
    for has_targets in 1 0; do
        for idx_iter in {1..100}; do
            ((i=i+1))
            if [[ "$DRY_RUN" == "1" ]]; then
                continue
            fi
            if [[ $i -eq $PBS_ARRAY_INDEX ]]; then
                export SINGULARITYENV_copula_type=$copula_type
                export SINGULARITYENV_has_targets=$has_targets
                export SINGULARITYENV_idx_iter=$idx_iter
                
                singularity exec --containall \
                    -B $ROOT_DIR:$ROOT_DIR \
                    -B $TMPDIR:/tmp \
                    $SIF_PATH \
                    bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate synthia-paper && cd $workdir && \
                    PYTHONHASHSEED=0 jupyter nbconvert --to html --execute stats.ipynb \
                    --output=$output_dir/stats-copula_type=$copula_type-has_targets=$has_targets-idx_iter=$idx_iter \
                    --ExecutePreprocessor.timeout=259200 --ExecutePreprocessor.iopub_timeout=300"

                exit 0
            fi
        done
    done
done
if [[ "$DRY_RUN" == "1" ]]; then
    echo "#PBS -J 1-$i"
fi
