#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G

cd /home/slurm/comp597/students/mreil2
git lfs install
git clone https://huggingface.co/datasets/InfImagine/FakeImageDataset
cd FakeImageDataset
git lfs pull
