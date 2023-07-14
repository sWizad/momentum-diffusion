# Diffusion Sampling with Momentum
The implementation for Diffusion Sampling with Momentum for Mitigating Divergence Artifacts (2023)

## Requirements

### Environments
Due to technical issues during experiments, we were unable to install Pytorch and Tensorflow in the same environment. So, we decided to create separate environments:
    - ```environment_default.yml``` (main)
    - ```environment_eval_DiT.yml``` (used for evaluating results in DiT experiment only)

To install requirement:

```bash
conda env create -f environment_default.yml
conda env create -f environment_eval_DiT.yml
```

If you encounter difficulties using GPU in Tensorflow, please refer to [the official website](https://www.tensorflow.org/install/pip).

### Datasets
- To reproduce results in DiT experiments, download OpenAI's ImageNet 256x256 reference batch from [here](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) and place it in ```./DiT/data/```.

- To re-compute channel-wise mean and standard deviation on COCO dataset, download the following dataset:

```bash
wget http://images.cocodataset.org/zips/train2014.zip
unzip train2014.zip
```

## DiT Experiments

To reproduce results in DiT experiments in the paper, run these commands:
```bash
cd ./DiT
bash run_dit_experiment.sh # generate images using different sampling methods
bash run_dit_evaluator.sh  # compute FID and other metrics
python utils.py --dir ../DiT_results
```
Do NOT forget to use ```environment_eval_DiT.yml``` when running the evaluator.

To train the model(s) in the paper, run this command:

```train
python train.py --input-data <path_to_data> --alpha 10 --beta 20
```

## Stable Diffusion Experiments
To reproduce results in DiT experiments in the paper, refer to commands in the relevant sections below. The results are primarily saved a csv file, which can be visualized using ```matplotlib``` and ```seaborn``` library. We provide code for generating plots in paper for reference in ```stable_diffusion/helper/visualizer.py```.

### Latent statistics on COCO
```bash
cd ./stable_diffusion
python compute_latents_stat.py --input_dir <path_to_dataset> --output_dir <path_to_save_latents> --device "cuda:0" > latent_stat.log
```
Results are available at ```latent_stat.log```.

### Convergence speed
```bash
cd ./stable_diffusion

# For GHVB
python generate.py --config_file configs/convergence_speed/ghvb.yml --device "cuda:0"
python evaluate.py --output_dir ../SD_output/SOTA_convergence_main --save_dir ../SD_results/SOTA_convergence_main --mode all --folder_gt "ghvb"

# For PLMS w/ HB and NT
python generate.py --config_file configs/convergence_speed/hb.yml --device "cuda:0"
python generate.py --config_file configs/convergence_speed/nesterov.yml --device "cuda:0"
python evaluate.py --output_dir ../SD_output/SOTA_convergence_aba --save_dir ../SD_results/SOTA_convergence_aba --mode all --folder_gt "hb"
```
Results are saved as csv files at ```../SD_results/SOTA_convergence_main``` and ```../SD_results/SOTA_convergence_aba```.


### Magnitude score
```bash
cd ./stable_diffusion

# sample images
python generate.py --config_file configs/magnitude_score/ghvb.yml --device "cuda:0"
python generate.py --config_file configs/magnitude_score/hb.yml --device "cuda:0"
python generate.py --config_file configs/magnitude_score/dpm.yml --device "cuda:0"

# compute evaluation metrics
python evaluate.py --output_dir ../SD_output/SOTA_artifacts --save_dir ../SD_results/SOTA_artifacts --mode all --folder_gt "ghvb"
```
Results are saved as csv files at ```../SD_results/SOTA_artifacts```.