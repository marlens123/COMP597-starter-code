#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

### run ResNet152 with CodeCarbon tracking 3x and for different batch sizes
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 2 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 3 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

    ${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 4 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 5 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 6 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

    ${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 7 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 8 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 1e-6 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 9 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs'