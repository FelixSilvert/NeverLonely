import random as r
import cv2
from controlnet_aux.pidi import PidiNetDetector
import uuid
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint
import torch
import requests
from PIL import Image
from io import BytesIO
from matplotlib import pyplot as plt

# We'll be exploring a number of pipelines today!
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionInpaintPipeline,
    StableDiffusionDepth2ImgPipeline
)

from diffusers import StableDiffusionPipeline

# We'll use a couple of demo images later in the notebook


def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content)).convert("RGB")


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid


# Set device
device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

txt2image_pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", orch_dtype=torch.float16, variant="fp16", use_safetensors=True).to(device)

model_id = "stabilityai/stable-diffusion-2-1-base"
img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id, variant="fp16", use_safetensors=True).to(device)


def without_init_image(prompt1, promt2):
    num_images = 1
    prompt = [prompt1] * num_images
    seed = 11235453
    generator = torch.Generator("cuda").manual_seed(seed)

    images = txt2image_pipe(prompt, generator=generator).images
    init_image = images[0]

    num_images = 3
    prompt = [promt2] * num_images
    seed = 11235453
    generator = torch.Generator("cuda").manual_seed(seed)

    images = img2img_pipe(prompt, image=init_image,
                          generator=generator, strength=0.65).images

    grid = image_grid(images, rows=1, cols=3)
    return grid


def with_image(init_image, text):

    seed = 46848
    generator = torch.Generator("cuda").manual_seed(seed)

    # Apply Img2Img
    result_image = img2img_pipe(
        prompt=text,
        image=init_image,  # The starting image
        generator=generator,
        strength=0.7,  # 0 for no change, 1.0 for max strength
    ).images[0]

    return result_image


def iterative_image2image(init_image, text):

    prompt = text

    image = init_image
    images = []
    for i in range(3):
        image = img2img_pipe(prompt, image=image, strength=0.6).images[0]
        images.append(image)
    return images


def image_restoration_same_prompt(prompt, guidance_scale_list, strength_list, number_of_inference_list, seed_liste):
    images = []
    for i in range(len(guidance_scale_list)):
        guidance_scale = guidance_scale_list[i]
        strength = strength_list[i]
        number_of_inference = number_of_inference_list[i]
        seed = seed_liste[i]
        num_images = 1
        generator = torch.Generator("cuda").manual_seed(seed)

        images = txt2image_pipe(prompt, guidance_scale=guidance_scale, strength=strength,
                                generator=generator, num_inference_steps=number_of_inference).images
        init_image = images[0]
        images.append(init_image)
    return images


def sketch2image_pidinet(init_image):
    preprocessor = PidiNetDetector.from_pretrained(
        "lllyasviel/Annotators").to("cuda")

    image_preprocessed = preprocessor(
        init_image,
        detect_resolution=512,
        image_resolution=512,
        apply_filter=True).convert('RGB').resize((512, 512))
    return image_preprocessed


def sketch2image_color_filtering(init_image):
    init_image.save("temp.png")
    img2 = cv2.imread("temp.png")
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    image_preprocessed = Image.fromarray(bw_image).convert("RGB")

    return image_preprocessed


def object_removal(init_image, element_to_remove):

    seed = 46848
    generator = torch.Generator("cuda").manual_seed(seed)

    result_image = img2img_pipe(
        prompt="Remove the"+element_to_remove,
        image=init_image,
        generator=generator,
        strength=0.7,
    ).images[0]

    return result_image


def object_adding(init_image, element_to_add):

    seed = 46848
    generator = torch.Generator("cuda").manual_seed(seed)

    result_image = img2img_pipe(
        prompt="add the"+element_to_add+"in the picture",
        image=init_image,
        generator=generator,
        strength=0.6,
    ).images[0]

    return result_image


def surpriseCharacter():
    sex = ["male", "women"]
    hair = ["blond", "brown", "black", "redhead"]
    eyes = ["blue", "brown", "yellow", "green", "grey", "black"]
    age = ["minor", "adult", "old"]
    skinColor = ["white", "black", "asian", "arabian", "indian"]
    hairStyle = ["straight", "curly", "wavy"]

    val1 = r.randint(0, 1)
    val2 = r.randint(0, 3)
    val3 = r.randint(0, 5)
    val4 = r.randint(0, 2)
    val5 = r.randint(0, 4)
    val6 = r.randint(0, 2)

    return skinColor[val5] + " " + age[val4] + " " + sex[val1] + ", with " + hair[val2] + " " + hairStyle[val6] + " " + "hair," + " " + eyes[val3] + " " + "eyes"
