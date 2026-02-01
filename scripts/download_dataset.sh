#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --error=error.txt

# environment setup
SCRIPTS_DIR=${COMP597_SLURM_SCRIPTS_DIR:-$(readlink -f -n $(dirname $0))}
. ${SCRIPTS_DIR}/conda_init.sh
conda activate ${COMP597_JOB_CONDA_ENV_PREFIX}

cd /mnt/teaching/slurm/mreil2

hf download InfImagine/FakeImageDataset \
  --repo-type dataset \
  --include "ImageData/val/SDv21-CC15K/*"