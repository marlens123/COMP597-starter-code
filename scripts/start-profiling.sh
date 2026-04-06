#!/bin/bash

SCRIPTS_DIR=$(readlink -f -n $(dirname $0))
REPO_DIR=$(readlink -f -n ${SCRIPTS_DIR}/..)

# NOTE: All these are number of workers = 0

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_time_stats \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_time_stats \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_time_stats \
#    --seed 42

### run ResNet152 End-to-end Resource Trainer
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 42

## run ResNet152 Basic Resource Trainer with different batch sizes
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 42

### run ResNet152 with CodeCarbon tracking 3x and for different batch sizes
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats codecarbon_resnet \
#    --trainer_stats_configs.codecarbon.run_num 1 \
#    --trainer_stats_configs.codecarbon.project_name test \
#    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats codecarbon_resnet \
#    --trainer_stats_configs.codecarbon.run_num 2 \
#    --trainer_stats_configs.codecarbon.project_name test \
#    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats codecarbon_resnet \
#    --trainer_stats_configs.codecarbon.run_num 3 \
#    --trainer_stats_configs.codecarbon.project_name test \
#    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
#    --seed 42

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \#
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_time_stats \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \#
#    --trainer_stats end_to_end_time_stats \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_time_stats \
#    --seed 84

## run ResNet152 End-to-end Resource Trainer
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats end_to_end_energy_stats \
#    --seed 84

## run ResNet152 Basic Resource Trainer with different batch sizes
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 128 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats basic_resources_stats \
#    --trainer_configs.basic_resources.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/basic_resources_logs' \
#    --seed 84

## run ResNet152 with CodeCarbon tracking 3x and for different batch sizes
#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 32 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats codecarbon_resnet \
#    --trainer_stats_configs.codecarbon.run_num 1 \
#    --trainer_stats_configs.codecarbon.project_name test \
#    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
#    --seed 84

#${SCRIPTS_DIR}/srun.sh \
#    --logging.level INFO \
#    --model resnet152 \
#    --data fakeimagenet \
#    --trainer resnet_simple \
#    --batch_size 64 \
#    --learning_rate 0.01 \
#    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
#    --trainer_stats codecarbon_resnet \
#    --trainer_stats_configs.codecarbon.run_num 2 \
#    --trainer_stats_configs.codecarbon.project_name test \
#    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
#    --seed 84

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 3 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
    --seed 84

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_time_stats \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_time_stats \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_time_stats \
    --seed 120

## run ResNet152 End-to-end Resource Trainer
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_energy_stats \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_energy_stats \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats end_to_end_energy_stats \
    --seed 120

## run ResNet152 Basic Resource Trainer with different batch sizes
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

## run ResNet152 with CodeCarbon tracking 3x and for different batch sizes
${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 32 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 1 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 64 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 2 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
    --seed 120

${SCRIPTS_DIR}/srun.sh \
    --logging.level INFO \
    --model resnet152 \
    --data fakeimagenet \
    --trainer resnet_simple \
    --batch_size 128 \
    --learning_rate 0.01 \
    --data_configs.fakeimagenet.folder '${COMP597_JOB_STUDENT_STORAGE_DIR}/fakeimagenet/FakeImageNet/train' \
    --trainer_stats codecarbon_resnet \
    --trainer_stats_configs.codecarbon.run_num 3 \
    --trainer_stats_configs.codecarbon.project_name test \
    --trainer_stats_configs.codecarbon.output_dir '${COMP597_JOB_STUDENT_STORAGE_DIR}/resnet/codecarbonlogs' \
    --seed 120