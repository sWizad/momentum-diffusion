import argparse
import os
import time
from copy import copy
from pathlib import Path

import pandas as pd
import torch
import yaml
from diffusers import DPMSolverMultistepScheduler, EulerDiscreteScheduler
from easydict import EasyDict as edict
from tqdm.auto import tqdm

from diffusion.momentum_scheduler import MomentumDPMSolverMultistepScheduler
from diffusion.pipeline import CustomPipeline
from diffusion.utils import save_prediction

def main(args, prompt, prompt_id, device, mode):
    """
    Generate images using Stable Diffusion pipeline from diffusers.
    Iterate over #sampling_steps x random_seed for a given prompt.
    """
    assert mode in ["gt", "proposed"]

    # Declare Stable Diffusion pipeline
    method = args.scheduler.method
    pipe = CustomPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.scheduler = globals()[args.scheduler.base].from_config(pipe.scheduler.config)
    if method in ["hb", "ghvb", "nt"]:
        pipe.init_scheduler(args.scheduler.method, args.scheduler.order)

    # Generate images
    results = []
    with tqdm(total=len(args.steps) * args.image_per_case) as pbar:
        for step in args.steps:
            for i in range(args.image_per_case):
                # reset scheduler
                seed = args.init_seed + i
                generator = torch.Generator(device=device).manual_seed(seed)
                if method in ["hb", "ghvb", "nt"]:
                    pipe.clear_scheduler()
                elif method in ["dpm"]:
                    pipe.scheduler.initialize_momentum(beta=args.scheduler.momentum)
                
                # sample image
                start_time = time.time()
                image, latents = pipe.generate({
                    "prompt": prompt,
                    "num_inference_steps": step,
                    "guidance_scale": args.guidance_scale,
                    "generator": generator,
                })
                elapsed_time = time.time() - start_time

                # determine save path for image and latent vector
                if method in ["hb", "ghvb", "nt"]:
                    order = str(args.scheduler.order).replace(".", "_")
                elif method in ["dpm"]:
                    order = str(args.scheduler.momentum).replace(".", "_")

                filename = f"{order}/{step}/prompt-{prompt_id}_seed-{seed}" 
                img_path = os.path.join(args.save_dir, "images", f"{filename}.png")
                latent_path = os.path.join(args.save_dir, "latents", f"{filename}.pt")

                gt_img_path = img_path.replace(f"{step}/", "gt/")
                gt_latent_path = latent_path.replace(f"{step}/", "gt/")

                # save image and latent vector
                if mode == "gt":
                    save_prediction(image, gt_img_path, latents, gt_latent_path)
                elif mode == "proposed":
                    save_prediction(image, img_path, latents, latent_path)
                    result = {
                        "step": step,
                        "order": order,
                        "prompt": prompt_id,
                        "seed": seed,
                        "img_path": img_path,
                        "latent_path": latent_path,
                        "gt_img_path": gt_img_path, 
                        "gt_latent_path": gt_latent_path,
                        "elapsed_time": elapsed_time,
                    }
                    results.append(result)
                
                pbar.update(1)
    
    # save log as csv file
    if mode == "proposed":
        filename = f"log_prompt-{prompt_id}_order-{order}.csv"
        save_path = os.path.join(args.save_dir, filename)
        df = pd.DataFrame.from_dict(results)
        df.to_csv(save_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, required=True, help='path to config file')
    parser.add_argument('--prompts_id_start', type=int, default=0, help='start index of prompt')
    parser.add_argument('--prompts_id_end', type=int, default=None, help='end index of prompt')
    parser.add_argument('--device', type=str, required=True, help='gpu id')
    cli = parser.parse_args()

    with open(cli.config_file, 'r') as f:
        args = yaml.safe_load(f)
    args = edict(args)
    
    # select prompts
    all_prompts = args.prompts
    start = cli.prompts_id_start
    end = cli.prompts_id_end if cli.prompts_id_end is not None else len(all_prompts)
    all_prompts = all_prompts[start:end]
    
    method = args.scheduler.method
    assert method in ["hb", "ghvb", "nt", "dpm"]
    momentum_type = "order" if method in ["hb", "ghvb", "nt"] else "momentum"
    
    # generate samples
    if "proposed" in args.mode:
        momentum_numbers = getattr(args.scheduler, momentum_type)
        for i, prompt in enumerate(all_prompts):
            prompt_id = cli.prompts_id_start + i
            for mm_number in momentum_numbers:
                new_args = copy(args)
                setattr(new_args.scheduler, momentum_type, mm_number)
                main(new_args, prompt, prompt_id, cli.device, "proposed")

    # generate ground-truth
    if "gt" in args.mode:
        momentum_numbers = [4.0]
        for i, prompt in enumerate(all_prompts):
            prompt_id = cli.prompts_id_start + i
            for mm_number in momentum_numbers:
                new_args = copy(args)
                setattr(new_args.scheduler, momentum_type, mm_number)
                setattr(new_args, "steps", [1000])
                main(new_args, prompt, prompt_id, cli.device, "gt")