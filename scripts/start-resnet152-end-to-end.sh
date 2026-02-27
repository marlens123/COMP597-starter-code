#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)


### run ResNet152 End-to-end Resource Trainer
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_stats \