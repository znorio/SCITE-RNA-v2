#!/usr/bin/env bash

for i in $(seq 0 10); do
  sbatch \
    --job-name="sc${i}" \
    --mem-per-cpu=32G \
    --time=20:00:00 \
    --output="logs/slurm_sc${i}.out" \
    --error="logs/slurm_sc${i}.err" \
    --wrap="Rscript generate_results_python_r/comparison_clones_sclineager_dendro_sciterna.R --test=${i}"
done

