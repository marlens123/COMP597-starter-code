#!/bin/bash
# Wrapper for dataset generation

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))

COMP597_JOB_COMMAND="python generate_data.py" ${SCRIPTS_DIR}/srun.sh "$@"
