#!/usr/bin/env bash

set -e

if [ ! -d src ]; then
    echo "Please run this script from the root folder"
    exit 1
fi

source singularity/singexec.sh

singexec "python src/plot.py"
