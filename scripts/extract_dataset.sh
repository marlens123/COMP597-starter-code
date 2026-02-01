#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --mem=8G

cd /mnt/teaching/slurm/mreil2/FakeImageDataset/ImageData

cd /home/slurm/comp597/students/mreil2/FakeImageDataset/ImageData

# Find all .tar.gz.01 files (first part of each archive)
find . -type f -name "*.tar.gz.01" | while read first; do
    # Get the directory and base name
    dir=$(dirname "$first")
    base=$(basename "$first" .01)  # removes the .01

    echo "Processing $first -> $dir/$base.tar.gz"

    # Merge all parts in numeric order
    cat "$dir/$base".* > "$dir/$base.tar.gz"

    # Extract
    tar -xvf "$dir/$base.tar.gz" -C "$dir"

    # Optional: remove the combined tar.gz to save space
    # rm "$dir/$base.tar.gz"
done

