import glob
import os
from pathlib import Path

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image

pil_to_tensor = transforms.Compose([
    transforms.PILToTensor()
])

def save_prediction(
    img=None, img_path=None, latent=None, latent_path=None
):
    if (img is not None) and (img_path is not None):
        os.makedirs(Path(img_path).parent, exist_ok=True)
        img.save(img_path)
    
    if (latent is not None) and (latent_path is not None):
        os.makedirs(Path(latent_path).parent, exist_ok=True)
        torch.save(latent, latent_path)

def combine_log(glob_name, save_result=False):
    df = []
    for log_path in glob.glob(glob_name):
        temp = pd.read_csv(log_path)
        df.append(temp)
    
    df = pd.concat(df)
    df = df.sort_values(by=["order", "prompt", "seed", "step"]).reset_index(drop=True)
    return df

def image_as_tensor(path):
    img = Image.open(path)
    img = pil_to_tensor(img)
    return img

def latent_as_tensor(path):
    latent = torch.load(path, map_location="cpu")
    return latent