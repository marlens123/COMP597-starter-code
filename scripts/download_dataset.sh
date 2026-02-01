#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G
#SBATCH --error=error.txt

conda activate /home/slurm/comp597/conda/envs/comp597
cd /mnt/teaching/slurm/mreil2

hf download InfImagine/FakeImageDataset \
  --repo-type dataset \
  --include "ImageData/val/SDv21-CC15K/*"