#### This is normal generation
import torch
from diffusion.pipeline import CustomPipeline

order = {"p": 1.5, "c": 1.5}
pipe = CustomPipeline.from_pretrained("runwayml/stable-diffusion-v1-5").to("cuda:3")
pipe.init_scheduler(method='UniPC', order=order) #PLMS_HB, PLMS_NT, GHVB, DPM-Solver++, UniPC

PROMPT = 'astronaut on bicycle'

GUIDANCE = 7
INIT_SEED = 1024
for idx in range(9):
    generator = torch.Generator(device='cuda:3').manual_seed(INIT_SEED + 1*idx)
    image = pipe(prompt = PROMPT, 
        num_inference_steps=20, guidance_scale=GUIDANCE,
        generator=generator,).images[0]
    image.save(f'output-std-{idx}.png')
    if hasattr(pipe,'clear_scheduler'): pipe.clear_scheduler()