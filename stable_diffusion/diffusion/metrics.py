import lpips
import numpy as np
import torch
import torch.nn as nn
from torchmetrics import PeakSignalNoiseRatio

from .utils import image_as_tensor, latent_as_tensor

MEANS = torch.tensor([0.1455953, -0.10772908, 0.023680864, 0.045954444]).view( 4, 1, 1)
STDS = torch.tensor([0.9220753, 1.0913924, 0.73026365, 0.72270846]).view(4, 1, 1)
MEAN = 0.02687545
STD = 0.8845245

DEVICE = torch.device("cuda")
PSNR = PeakSignalNoiseRatio(data_range=255., reduction="elementwise_mean")
loss_fn = lpips.LPIPS(net='vgg').to(DEVICE)

def calculate_psnr(pred_path, gt_path):
    pred = image_as_tensor(pred_path)
    gt = image_as_tensor(gt_path)
    assert pred.shape == gt.shape

    result = PSNR(pred,gt).item()
    return result

def calculate_lpips(pred_path, gt_path):
    pred = image_as_tensor(pred_path).to(DEVICE)
    gt = image_as_tensor(gt_path).to(DEVICE)
    assert pred.shape == gt.shape

    result = loss_fn(pred,gt).item()
    return result

def calculate_l1_norm(pred_path, gt_path, reduce):
    assert reduce in ["sum", "mean"]
    pred = latent_as_tensor(pred_path)
    gt = latent_as_tensor(gt_path)
    assert pred.shape == gt.shape

    result = getattr(torch, reduce)(torch.abs(pred - gt)).item()
    return result

def calculate_l2_norm(pred_path, gt_path, reduce):
    assert reduce in ["sum", "mean"]
    pred = latent_as_tensor(pred_path)
    gt = latent_as_tensor(gt_path)
    assert pred.shape == gt.shape

    result = getattr(torch, reduce)((pred - gt)**2).item()
    return result

def calculate_order_of_convergence(
    df,
    selected_steps=[10, 20, 40, 80, 160, 320, 640],
):
    selected_df = df[df["step"].isin(selected_steps)].reset_index(drop=True)
    num_unique_step = len(selected_df["step"].unique())
    order_df = selected_df.copy()

    # calculate e
    order_df = order_df.rename(columns={"l1_norm": "e_new"})
    order_df["e_old"] = order_df["e_new"].shift()

    # calculate h
    order_df["h_new"] = 1000 / order_df["step"]
    order_df["h_old"] = order_df["h_new"].shift()

    # calculate numerical order of convergence
    order_df = order_df[order_df.index % num_unique_step != 0]
    order_df["q"] = np.log2(order_df["e_new"] / order_df["e_old"]) / np.log2(order_df["h_new"] / order_df["h_old"])
    
    return order_df

def preprocess_latents(latents, per_channel_stat=True, kernel_size=4):
    # normalize
    if not per_channel_stat:
        latents = (latents - MEAN) / STD
    else:
        latents = (latents - MEANS) / STDS
 
    # take absolute, depth-wise max, and max pooling
    KERNEL_SIZE = kernel_size
    maxpool = nn.MaxPool2d(KERNEL_SIZE)
 
    latents = torch.abs(latents)
    max_latents = torch.max(latents, axis=0)[0].unsqueeze(0)
    max_latents = maxpool(max_latents).squeeze().numpy()
 
    return max_latents

def calculate_artifact_score(
    latent_path,
    per_channel_stat=True,
    kernel_size=4,
    threshold=3,
):
    latents = latent_as_tensor(latent_path)
    kwargs = {"per_channel_stat": per_channel_stat, "kernel_size": kernel_size}
    normalized_latents = preprocess_latents(latents.squeeze(), **kwargs)

    # calculate score
    normalized_latents[normalized_latents < threshold] = 0
    score = np.mean(normalized_latents)
    return score