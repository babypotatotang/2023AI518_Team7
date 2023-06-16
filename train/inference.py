from datasets import load_from_disk 
from diffusers import StableDiffusionPipeline
import torch

ds = load_from_disk('/home2/s20235025/Melon_test')

model_ft = "/home2/s20235025/Melon_ai518/epoch35" # 35 epoch 
pipe_ft = StableDiffusionPipeline.from_pretrained(model_ft, torch_dtype=torch.float16)
pipe_ft.to("cuda")
pipe_ft.safety_checker = None
pipe_ft.requires_safety_checker = False

model_sd = "CompVis/stable-diffusion-v1-4"
pipe_sd = StableDiffusionPipeline.from_pretrained(model_sd, torch_dtype=torch.float16)
pipe_sd.to("cuda")
pipe_sd.safety_checker = None
pipe_sd.requires_safety_checker = False

for index, prompt in enumerate(ds['text']):
    image = pipe_ft(prompt, num_inference_steps=50, guidance_scale=8).images[0]
    image.save(f"/home2/s20235025/Melon_ai518/output/ft/{index:04d}.png")    

    image = pipe_sd(prompt, num_inference_steps=50, guidance_scale=8).images[0]
    image.save(f"/home2/s20235025/Melon_ai518/output/sd/{index:04d}.png")    
    