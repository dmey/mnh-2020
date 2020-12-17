#!/usr/bin/env bash

#PBS -lwalltime=72:00:00
#PBS -lselect=1:ncpus=32:mem=62gb
#PBS -J 1-180

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
#for USE_ZENITH in 0 1; do
for copula_type in parametric tll gaussian; do
    for data_fraction in 1.0; do
        for factor_synthetic in 10 5 1; do
            for has_targets in 1 0 ; do
                for idx_iter_synthetic in 0 1 2 3 4 5 6 7 8 9; do
                    ((i=i+1))
                    if [[ "$DRY_RUN" == "1" ]]; then
                        continue
                    fi
                    if [[ $i -eq $PBS_ARRAY_INDEX ]]; then
                        export SINGULARITYENV_factor_synthetic=$factor_synthetic
                        export SINGULARITYENV_idx_iter_synthetic=$idx_iter_synthetic
                        export SINGULARITYENV_copula_type=$copula_type
                        export SINGULARITYENV_has_targets=$has_targets
                        export SINGULARITYENV_data_fraction=$data_fraction

                        # https://github.com/MPAS-Dev/MPAS-Analysis/issues/407#issuecomment-436383906
                        export SINGULARITYENV_HDF5_USE_FILE_LOCKING=FALSE
                        
                        singularity exec --containall \
                            -B $ROOT_DIR:$ROOT_DIR \
                            -B $TMPDIR:/tmp \
                            $SIF_PATH \
                            bash -c ". /miniconda/etc/profile.d/conda.sh && conda activate synthia-paper && cd $workdir && \
                            PYTHONHASHSEED=0 jupyter nbconvert --to html --execute ml_synth.ipynb \
                            --output=$output_dir/ml-copula_type=$copula_type-has_targets=$has_targets-data_fraction=$data_fraction-factor_synthetic=$factor_synthetic-idx_iter_synthetic=$idx_iter_synthetic \
                            --ExecutePreprocessor.timeout=259200 --ExecutePreprocessor.iopub_timeout=300"
                        
                        exit 0
                    fi
                done
            done
        done
    done
done
if [[ "$DRY_RUN" == "1" ]]; then
    echo "#PBS -J 1-$i"
fi
