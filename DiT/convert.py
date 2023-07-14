from tqdm.auto import tqdm
from PIL import Image
import numpy as np
import argparse

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path

parser = argparse.ArgumentParser()
parser.add_argument("--sample_folder_dir", type=str, required=True)
parser.add_argument("--num_fid_samples",  type=int, required=True)
args = parser.parse_args()

create_npz_from_sample_folder(args.sample_folder_dir, args.num_fid_samples)