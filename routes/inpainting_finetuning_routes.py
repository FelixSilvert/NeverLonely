import uuid
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint


# Create a Blueprint for imageToImage routes
inpainting_finetuning_routes = Blueprint(
    'inpainting_finetuning_routes', __name__)

# Define the routes for the Blueprint


def get_param(param_name, default_value=None):
    value = request.args.get(param_name)
    if not value:
        return default_value
    return value


@inpainting_finetuning_routes.route('/', methods=['GET'])
def inpainting_finetuning():
    pass
