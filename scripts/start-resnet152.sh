#!/bin/bash

# ==============================
# Run file for GPT2 example - how to use the launch script to run GPT2 with simple trainer
# ==============================

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run GPT2 Simple Trainer
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats simple

### run GPT2 with CodeCarbon tracking
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 4 \
    --learning_rate 1e-6 \
    --trainer_stats codecarbon \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/gpt2-example/codecarbonlogs'
