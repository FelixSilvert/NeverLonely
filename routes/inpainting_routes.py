import uuid
from utils.inpainting_utils import inpaiting_model, pipe, object_addition_with_instruct_inpainting, object_removal_with_instruct_inpainting
from flask import Blueprint, request, send_file
from random import randint


# Create a Blueprint for imageToImage routes
inpainting_routes = Blueprint('inpainting_routes', __name__)

# Define the routes for the Blueprint


def get_param(param_name, default_value=None):
    value = request.args.get(param_name)
    if not value:
        return default_value
    return value


@inpainting_routes.route('/', methods=['GET'])
def inpainting():
    filename = f'image_{uuid.uuid4().hex}.png'
    init_image = get_param('init_image', "")
    if init_image == "":
        return "Please provide init_image"
    mask_prompt = get_param("mask_prompt", "")
    prompt = get_param("prompt", "")

    inpaiting_model(init_image, mask_prompt, prompt).save(
        "./content/"+filename)

    return send_file("./content/"+filename, mimetype="img/png")


@inpainting_routes.route('/manualy', methods=['GET'])
def inpainting_manualy():
    filename = f'image_{uuid.uuid4().hex}.png'
    mask_image = get_param('mask_image', "")
    init_image = get_param('init_image', "")
    negative_prompt = get_param('negative_prompt', "")
    fitting_degree = get_param('fitting_degree', 1)
    num_inference_steps = get_param('num_inference_steps', 50)
    guidance_scale = get_param('guidance_scale', 12)
    instruction = get_param('instruction', "add")
    if init_image == "":
        return "Please provide init_image"
    if instruction == "add":
        object_addition_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt, fitting_degree,
                                                 num_inference_steps, guidance_scale).save(
            "./content/"+filename)
    elif instruction == "remove":
        object_removal_with_instruct_inpainting(pipe, init_image, mask_image, negative_prompt, fitting_degree,
                                                num_inference_steps, guidance_scale).save(
            "./content/"+filename)
    else:
        return "Please provide instruction"
    return send_file("./content/"+filename, mimetype="img/png")
