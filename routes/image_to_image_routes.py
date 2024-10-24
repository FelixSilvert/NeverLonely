import uuid
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint


# Create a Blueprint for imageToImage routes
imageToImage_routes = Blueprint('imageToImage_routes', __name__)

# Define the routes for the Blueprint


def get_param(param_name, default_value=None):
    value = request.args.get(param_name)
    if not value:
        return default_value
    return value


@imageToImage_routes.route('/', methods=['GET'])
def image_to_image():
    pass
