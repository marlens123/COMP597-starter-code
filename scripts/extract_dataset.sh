#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G

cd /home/slurm/comp597/students/mreil2/FakeImageDataset/ImageData

# Recursively find all split tar.gz parts, combine and extract
find . -type f -name "*.tar.gz.*" | while read f; do
    # Get the base filename (remove .part numbers)
    base="${f%%.tar.gz.*}.tar.gz"
    echo "Processing $f -> $base"

    # Combine all parts into one tar.gz
    cat "${f%%.*}".tar.gz.* > "$base"

    # Extract
    tar -xvf "$base"

    # Remove the combined tar.gz to save space
    rm "$base"
done