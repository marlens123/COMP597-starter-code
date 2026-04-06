#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)


### run ResNet152 Basic Resource Trainer 3x and with different batch sizes
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 42

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 42

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 42

    ${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 84

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 84

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 84

    ${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats basic_resources_stats \
    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
    --seed 120