import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm

from diffusion.utils import save_prediction

transform = transforms.Compose([
    transforms.Resize(512),
    transforms.CenterCrop(512),
    transforms.ToTensor(),
])

def main(args):
    device = torch.device(args.device)

    pipe = StableDiffusionPipeline.from_pretrained(args.model_id, torch_dtype=torch.float32)
    vae = pipe.vae.to(device)

    glob_name = f"{args.input_dir}/*.jpg"
    paths = list(glob.glob(glob_name))
    
    # 1. prepare latents
    with torch.no_grad():
        for path in tqdm(paths):
            original_pil_img = Image.open(path).convert('RGB')

            # select only image with original resolution more than (512, 512)
            if (original_pil_img.size[0] < 512) or (original_pil_img.size[1] < 512):
                continue

            original_image = transform(original_pil_img).to(device)
            original_image = original_image.unsqueeze(0)

            # encode
            latents = vae.encode(original_image*2 - 1)
            latents = vae.config.scaling_factor * latents.latent_dist.sample()

            # save latents
            filename = Path(path).stem
            save_path = os.path.join(args.output_dir, f"{filename}.pt")
            save_prediction(latent=latents, latent_path=save_path)

    # 2. calculate statistics
    glob_name = os.path.join(args.output_dir, "*.pt")
    paths = glob.glob(glob_name)

    latents = []
    for path in tqdm(paths):
        latent = torch.load(path, map_location="cpu")
        assert latent.shape == (1, 4, 64, 64)
        latents.append(latent)

    latents = np.array(latents)
    data = torch.concat(list(latents)).numpy()

    MEAN = data.mean()
    STD = data.std()
    MEANS = []
    STDS = []
    for i in range(4):
        MEANS.append(data[:, i].mean())
        STDS.append(data[:, i].std())

    print(f"Global mean = {MEAN}")
    print(f"Global std = {STD}")
    print(f"Channel-wise mean = {MEANS}")
    print(f"Channel-wise std = {STDS}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=False, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument('--input_dir', type=str, required=True, help='path to input file')
    parser.add_argument('--output_dir', type=str, required=True, help='path to save latents')
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    main(args)