#!/bin/bash

run_experiment() {
  local mode="$1"
  local orders=("${@:2}")

  for order in "${orders[@]}"
  do
    for step in "${steps[@]}"
    do
      dir=$(echo "$order" | tr '.' '_')
      output_dir="../DiT_output/$mode/$dir/$step"
      torchrun --nnodes=1 --nproc_per_node=$num_gpu sample_ddp.py --model DiT-XL/2 \
      --num-fid-samples $num_samples --num-sampling-steps $step --sample-dir $output_dir \
      --mode "$mode" --order "$order" --per-proc-batch-size $batch_size
    done
  done
}

# Define common variables
num_gpu=4
num_samples=10000
batch_size=16
steps=(25 20 15 10 9 8 7 6)

# Run LTSP
mode="ltsp"
orders=(4.0)
run_experiment "$mode" "${orders[@]}"

# Run DPM-Solver++
mode="dpm"
orders=(2.0)
run_experiment "$mode" "${orders[@]}"

# Run GHVB
mode="ghvb"
orders=(3.9 3.8)
run_experiment "$mode" "${orders[@]}"

# Run PLMS w/ HB, DDIM, and PLMS4
mode="hb"
orders=(4.0 3.9 3.8 1.0)
run_experiment "$mode" "${orders[@]}"