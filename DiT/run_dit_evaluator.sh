#!/bin/bash

run_evaluator() {
  local mode="$1"
  local orders=("${@:2}")

  for order in "${orders[@]}"
  do
    for step in "${steps[@]}"
    do
      # create npz file
      order_reformat=$(echo "$order" | tr '.' '_')
      log_path="../DiT_results/$mode/$order_reformat/$step.log"
      pred_dir="../DiT_output/$mode/$order_reformat/$step/$prefix"
      pred_path="$pred_dir.npz"

      # calculate FID, Inception Score, Precision, Recall, and sFID
      mkdir -p "../DiT_results/$mode/$order_reformat"
      python convert.py --sample_folder_dir $pred_dir --num_fid_samples $num_samples
      python evaluator.py $ref_path $pred_path > $log_path
      rm $pred_path
    done
  done
}

# Define common variables
ref_path="./data/VIRTUAL_imagenet256_labeled.npz"
prefix="DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.5-seed-0"
num_samples=10000
steps=(25 20 15 10 9 8 7 6)

# Run LTSP
mode="ltsp"
orders=(4.0)
run_evaluator "$mode" "${orders[@]}"

# Run DPM-Solver++
mode="dpm"
orders=(2.0)
run_evaluator "$mode" "${orders[@]}"

# Run GHVB
mode="ghvb"
orders=(3.9 3.8)
run_evaluator "$mode" "${orders[@]}"

# Run PLMS w/ HB, DDIM, and PLMS4
mode="hb"
orders=(4.0 3.9 3.8 1.0)
run_evaluator "$mode" "${orders[@]}"