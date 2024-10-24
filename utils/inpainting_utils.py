from diffusers import UniPCMultistepScheduler
from powerpaint.utils.utils import TokenizerWrapper, add_tokens
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
    StableDiffusionPowerPaintBrushNetPipeline,
)
from powerpaint.models.BrushNet_CA import BrushNetModel
from diffusers import DPMSolverMultistepScheduler
from diffusers.utils import load_image
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image, ImageOps
import numpy as np
from safetensors.torch import load_model, save_model
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
import cv2
import matplotlib.pyplot as plt
import torch
import requests
import PIL
from io import BytesIO
from matplotlib import pyplot as plt
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
import requests
from transformers import AutoProcessor, Kosmos2ForConditionalGeneration
import random
import sys
import os
sys.path.insert(0, os.path.join("/content/PowerPaint"))


# from powerpaint.power_paint_tokenizer import PowerPaintTokenizer


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = PIL.Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols*w, i//cols*h))
    return grid

######
# @title Automatic Mask Generation with Textual Description
######


processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained(
    "CIDAS/clipseg-rd64-refined")


# 'mask_prompt' must have at least 2 keywords

def generate_mask_image(image, mask_prompt, out_fpath, debug=False):
    temp_fpath = f"temp.png"
    init_image = image
    if isinstance(mask_prompt, str):
        mask_prompt = [mask_prompt, mask_prompt]
    if isinstance(mask_prompt, list) and len(mask_prompt) == 1:
        mask_prompt = mask_prompt * 2
    inputs = processor(text=mask_prompt, images=[
                       init_image] * len(mask_prompt), padding="max_length", return_tensors="pt")

    # predict
    with torch.no_grad():
        outputs = model(**inputs)

    preds = outputs.logits.unsqueeze(1)

    # visualize prediction
    if debug:
        _, ax = plt.subplots(1, 5, figsize=(15, len(mask_prompt)+1))
        [a.axis('off') for a in ax.flatten()]
        ax[0].imshow(init_image)
        print(torch.sigmoid(preds[0][0]).shape)
        [ax[i+1].imshow(torch.sigmoid(preds[i][0]))
         for i in range(len(mask_prompt))]
        [ax[i+1].text(0, -15, prompts[i]) for i in range(len(mask_prompt))]

    plt.imsave(temp_fpath, torch.sigmoid(preds[1][0]))
    img2 = cv2.imread(temp_fpath)
    gray_image = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    (thresh, bw_image) = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)

    # fix color format
    cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)

    mask_image = PIL.Image.fromarray(bw_image)
    mask_image.save(out_fpath)
    return mask_image


device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

inpaiting_pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting")
inpaiting_pipe = inpaiting_pipe.to(device)

model2 = Kosmos2ForConditionalGeneration.from_pretrained(
    "microsoft/kosmos-2-patch14-224")
processor2 = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")


def inpaiting_model_mask_auto_with_mask_prompt(image, mask_prompt):
    out_fpath = f"mask.png"
    init_image = image
    mask_image = generate_mask_image(
        image, mask_prompt, out_fpath, debug=False)
    return mask_image


def inpaiting_model_mask_auto_without_mask_prompt(image):
    prompt = "<grounding> An image of"

    inputs = processor2(text=prompt, images=image, return_tensors="pt")

    generated_ids = model2.generate(
        pixel_values=inputs["pixel_values"],
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_embeds=None,
        image_embeds_position_mask=inputs["image_embeds_position_mask"],
        use_cache=True,
        max_new_tokens=64,
    )
    generated_text = processor2.batch_decode(
        generated_ids, skip_special_tokens=True)[0]
    processed_text = processor2.post_process_generation(
        generated_text, cleanup_and_extract=False)
    processed_text

    caption, entities = processor2.post_process_generation(generated_text)

    entiters = []

    for i in range(len(entities)):
        entiters.append(entities[i][0])

    print(entiters)

    init_image = image

    mask_prompt = [entiters[random.randint(0, len(entiters)-1)]]
    out_fpath = f"mask.png"
    mask_image = generate_mask_image(
        image, mask_prompt, out_fpath, debug=False)
    return mask_image


def inpaiting_model(image, mask_prompt, prompt):
    generator = torch.Generator("cuda")

    if mask_prompt == None:
        mask_image = inpaiting_model_mask_auto_without_mask_prompt(image)
    else:
        mask_image = inpaiting_model_mask_auto_with_mask_prompt(
            image, mask_prompt)

    result_image = inpaiting_pipe(
        prompt=prompt,
        image=image,  # The starting image
        mask_image=mask_image,
        generator=generator,
        strength=0.9,  # 0 for no change, 1.0 for max strength
    ).images[0]
    return result_image


checkpoint_dir = "/content/checkpoints"
local_files_only = True

# brushnet-based version
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="unet",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=False,
)
text_encoder_brushnet = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    subfolder="text_encoder",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=False,
)
brushnet = BrushNetModel.from_unet(unet)
base_model_path = os.path.join(checkpoint_dir, "realisticVisionV60B1_v51VAE")
pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
    base_model_path,
    brushnet=brushnet,
    text_encoder_brushnet=text_encoder_brushnet,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
    safety_checker=None,
)
pipe.unet = UNet2DConditionModel.from_pretrained(
    base_model_path,
    subfolder="unet",
    revision=None,
    torch_dtype=torch.float16,
    local_files_only=local_files_only,
)
pipe.tokenizer = TokenizerWrapper(
    from_pretrained=base_model_path,
    subfolder="tokenizer",
    revision=None,
    torch_type=torch.float16,
    local_files_only=local_files_only,
)

# add learned task tokens into the tokenizer
add_tokens(
    tokenizer=pipe.tokenizer,
    text_encoder=pipe.text_encoder_brushnet,
    placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
    initialize_tokens=["a", "a", "a"],
    num_vectors_per_token=10,
)
load_model(
    pipe.brushnet,
    os.path.join(checkpoint_dir,
                 "PowerPaint_Brushnet/diffusion_pytorch_model.safetensors"),
)

pipe.text_encoder_brushnet.load_state_dict(
    torch.load(os.path.join(checkpoint_dir, "PowerPaint_Brushnet/pytorch_model.bin")), strict=False
)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")


def task_to_prompt(control_type):
    if control_type == "object-removal":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    elif control_type == "context-aware":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = ""
        negative_promptB = ""
    elif control_type == "shape-guided":
        promptA = "P_shape"
        promptB = "P_ctxt"
        negative_promptA = "P_shape"
        negative_promptB = "P_ctxt"
    elif control_type == "image-outpainting":
        promptA = "P_ctxt"
        promptB = "P_ctxt"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"
    else:
        promptA = "P_obj"
        promptB = "P_obj"
        negative_promptA = "P_obj"
        negative_promptB = "P_obj"

    return promptA, promptB, negative_promptA, negative_promptB


@torch.inference_mode()
def predict(
    pipe,
    input_image,
    prompt,
    fitting_degree,
    ddim_steps,
    scale,
    negative_prompt,
    task,
):
    promptA, promptB, negative_promptA, negative_promptB = task_to_prompt(task)
    print(task, promptA, promptB, negative_promptA, negative_promptB)
    img = np.array(input_image["image"].convert("RGB"))

    W = int(np.shape(img)[0] - np.shape(img)[0] % 8)
    H = int(np.shape(img)[1] - np.shape(img)[1] % 8)
    input_image["image"] = input_image["image"].resize((H, W))
    input_image["mask"] = input_image["mask"].resize((H, W))

    np_inpimg = np.array(input_image["image"])
    np_inmask = np.array(input_image["mask"]) / 255.0

    np_inpimg = np_inpimg * (1 - np_inmask)

    input_image["image"] = PIL.Image.fromarray(
        np_inpimg.astype(np.uint8)).convert("RGB")

    result = pipe(
        promptA=promptA,
        promptB=promptB,
        promptU=prompt,
        tradoff=fitting_degree,
        tradoff_nag=fitting_degree,
        image=input_image["image"].convert("RGB"),
        mask=input_image["mask"].convert("RGB"),
        num_inference_steps=ddim_steps,
        brushnet_conditioning_scale=1.0,
        negative_promptA=negative_promptA,
        negative_promptB=negative_promptB,
        negative_promptU=negative_prompt,
        guidance_scale=scale,
        width=H,
        height=W,
    ).images[0]
    return result


def object_removal_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt, fitting_degree=1,
                                            num_inference_steps=50, guidance_scale=12):
    negative_prompt = negative_prompt + ", out of frame, lowres, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, disfigured, gross proportions, malformed limbs, watermark, signature"
    input_image = {"image": init_image, "mask": mask_image}
    image = predict(
        pipe,
        input_image,
        "empty scene blur",  # prompt
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        negative_prompt,
        "object-removal"  # task
    )
    return image


def object_addition_with_instruct_inpainting(pipe, init_image, mask_image, prompt, fitting_degree=1,
                                             num_inference_steps=50, guidance_scale=12):
    input_image = {"image": init_image, "mask": mask_image}
    image = predict(
        pipe,
        input_image,
        prompt,
        fitting_degree,
        num_inference_steps,
        guidance_scale,
        "",  # negative prompt
        "text-guided"  # task
    )
    return image
