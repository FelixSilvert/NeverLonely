from diffusers import StableDiffusionPipeline
import torch
from torchmetrics.functional.multimodal import clip_score
from functools import partial
import numpy as np

clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")

def calculate_clip_score(image, prompt):
    if isinstance(image, Image.Image):
        image = np.array(image) / 255.0
    else:
        raise ValueError("Not the good format")
    image_int = (image * 255).astype("uint8")
    clip_score_value = clip_score_fn(torch.from_numpy(image_int).permute(2, 0, 1).unsqueeze(0), [prompt]).detach()
    return round(float(clip_score_value), 4)

