
## Overview

This is the development repository for the paper "[Copula-based synthetic data generation for machine learning emulators in weather and climate: application to a simple radiation model](https://arxiv.org/abs/2012.09037)" simply meant as a placeholder.

For the paper's data archive that includes all model outputs as well as the [Singularity](https://sylabs.io/singularity/) image, please refer to https://doi.org/10.5281/zenodo.4320795 instead.

For the Python tool used to generate the synthetic data, please refer to the [Synthia repository](https://github.com/dmey/synthia).


## Requirements

- [Singularity](https://sylabs.io/singularity/) >= 3
- [Portable Batch System](https://en.wikipedia.org/wiki/Portable_Batch_System) (PBS) job scheduler*
- Today's high-performance computer (e.g. ~ 32 CPUs @ 2 500 MHz with 64 GB of RAM )

*Although PBS in not a strict requirement, it is required to run all helper scripts as included in this repository. Please note that depending on your specific system settings and resource availability, you may need to modify PBS parameters at the top of submit scripts stored in the `hpc` directory (e.g. `#PBS -lwalltime=72:00:00`).


## Initialization

Build the Singularity image with:

```
singularity build --remote singularity/image.sif singularity/image.def
```


## Usage

To reproduce the results from the experiments described in the paper, first fit all copula models to the reduced NWP-SAF dataset with:

```
qsub hpc/fit.sh
```

then, to generate synthetic data, run all machine learning model configurations, and compute the relevant statistics use:

```
qsub hpc/stats.sh
qsub hpc/ml_control.sh
qsub hpc/ml_synth.sh
```

Finally, to plot all artifacts included in the paper use:

```
qsub hpc/plot.sh
```


## Licence

Code released under [MIT license](./LICENSE.txt). Data from the reduced NWP-SAF dataset released under [CC BY 4.0](./data/LICENSE.txt).
