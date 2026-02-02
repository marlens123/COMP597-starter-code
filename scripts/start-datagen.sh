#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run GPT2 Simple Trainer
${SCRIPTS_DIR}/srun_datagen.sh \