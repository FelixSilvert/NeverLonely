import uuid

from utils.image_to_image_finetuning_utils import StableDiffusionModel
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint


# Create a Blueprint for imageToImage routes
image_to_image_finetuning_routes = Blueprint(
    'image_to_image_finetuning_routes', __name__)

# Define the routes for the Blueprint


def get_param(param_name, default_value=None):
    value = request.args.get(param_name)
    if not value:
        return default_value
    return value

# Path to the downloaded LoRA file
#lora_weights_path = "lora/skin_texture_style_sd1.5_v1.safetensors"
#lora_weights_path = "lora/Anamorphic_style_SD1.5.safetensors"
#lora_weights_path = "lora/perfection_style_SD1.5.safetensors"
#lora_weights_path = "lora/Tron_style_SD1.5.safetensors"
lora_weights_path = "lora/unreal_engine_style.safetensors"


# Initialize the model with LoRA
model = StableDiffusionModel(use_lora=True, lora_weights_path=lora_weights_path)

@image_to_image_finetuning_routes.route('/', methods=['GET'])
def textToImage():
    seed = get_param('seed', randint(0, 1000000))
    prompt = get_param('prompt', "")
    negative_prompt = get_param('negative_prompt', "")
    guidance_scale = get_param('guidance_scale', "")
    num_inference_steps = get_param('num_inference_steps', "")

    # Generate a unique filename using UUID
    filename = f'image_{uuid.uuid4().hex}.png'

    model.generate_text_to_image(
        negative_prompt=negative_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
        prompt=prompt
    ).save("./content/" + filename)

    return send_file("./content/" + filename, mimetype='image/png')
