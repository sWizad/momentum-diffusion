import argparse
from copy import copy
import glob
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from diffusion.metrics import (
    calculate_artifact_score,
    calculate_l1_norm,
    calculate_l2_norm,
    calculate_lpips,
    calculate_psnr,
    calculate_order_of_convergence,
)
from diffusion.utils import combine_log

tqdm.pandas()

def analyze(
    input_df,
    require_metrics=["l2_norm", "l1_norm", "lpips", "psnr", "artifact_score"],
):
    df = copy(input_df)

    available_metrics = {
        "l2_norm": lambda x: calculate_l2_norm(x.latent_path, x.gt_latent_path, reduce="mean") if os.path.exists(x.gt_latent_path) else np.nan,
        "l1_norm": lambda x: calculate_l1_norm(x.latent_path, x.gt_latent_path, reduce="mean") if os.path.exists(x.gt_latent_path) else np.nan,
        "lpips": lambda x: calculate_lpips(x.img_path, x.gt_img_path) if os.path.exists(x.gt_img_path) else np.nan,
        "psnr": lambda x: calculate_psnr(x.img_path, x.gt_img_path) if os.path.exists(x.gt_img_path) else np.nan,
        "artifact_score": lambda x: calculate_artifact_score(x.latent_path),
    }

    for metric in require_metrics:
        df[metric] = np.nan
        df[metric] = df.progress_apply(available_metrics[metric], axis=1)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, choices=["artifacts", "all"], required=True)
    parser.add_argument("--order_gt", type=float, default=4.0)
    parser.add_argument("--folder_gt", type=str, default="ghvb")
    args = parser.parse_args()

    ###################
    # 1. Prepare dataframes
    ###################
    dirs = glob.glob(os.path.join(args.output_dir, "*"))
    order_gt = str(args.order_gt).replace(".", "_")
    
    list_df = []
    for dir in dirs:
        glob_name = os.path.join(dir, "log_*.csv")
        df = combine_log(glob_name)

        # specify ground-truth path
        vfunc = lambda x: x.gt_img_path.replace(f"{x.order}/gt", f"{order_gt}/gt").replace(dir, f"{args.output_dir}/{args.folder_gt}")
        df["gt_img_path"] = df.apply(vfunc, axis=1)

        vfunc = lambda x: x.gt_latent_path.replace(f"{x.order}/gt", f"{order_gt}/gt").replace(dir, f"{args.output_dir}/{args.folder_gt}")
        df["gt_latent_path"] = df.apply(vfunc, axis=1)

        if "dpm" in dir:
            df["order"] = df["order"].apply(lambda x: "2" + x[1:])

        df["label"] = df["order"].apply(lambda x: float(x.replace("_", ".")))
        df["category"] = df["label"].apply(lambda x: "momentum" if int(x) != x else "original")
        if "hb" in dir:
            df["scheduler"] = df["label"].apply(lambda x: "DDIM w/ HB" if math.ceil(x) == 1.0 else f"PLMS{int(math.ceil(x))} w/ HB")
        elif "nesterov" in dir:
            df["scheduler"] = df["label"].apply(lambda x: "DDIM w/ NT" if math.ceil(x) == 1.0 else f"PLMS{int(math.ceil(x))} w/ NT")
        elif "ghvb" in dir:
            df["scheduler"] = "ghvb"
        elif "dpm" in dir:
            df["scheduler"] = "DPM-Solver++ w/ HB"

        list_df.append(df)
        
    df = pd.concat(list_df).reset_index(drop=True)

    ###################
    # 2. Calculate metrics and numerical order of Convergence
    ###################
    print("Calculating metrics...")
    os.makedirs(args.save_dir, exist_ok=True)
    if args.mode == "artifacts":
        df = analyze(df, require_metrics=["artifact_score"])
    elif args.mode == "all":
        df = analyze(df)
        selected_steps = df["step"].unique()
        order_df = calculate_order_of_convergence(df, selected_steps)

        save_path = os.path.join(args.save_dir, f"orderOfConvergence_{args.mode}-{args.order_gt}.log")
        order_df.to_csv(save_path, index=False)

    save_path = os.path.join(args.save_dir, f"metrics_{args.mode}-{args.order_gt}.log")
    df.to_csv(save_path, index=False)