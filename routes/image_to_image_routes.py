import uuid
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint
from utils.image_to_image_utils import without_init_image, surpriseCharacter, object_adding, object_removal, with_image


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
    filename = f'image_{uuid.uuid4().hex}.png'
    init_image = get_param('init_image', "")
    if init_image == "":
        return "Please provide init_image"
    prompt = get_param("prompt", "")
    instruction = get_param("instruction", "add")
    if instruction == "add":
        object_adding(init_image, prompt).save("./content/"+filename)
    elif instruction == "remove":
        object_removal(init_image, prompt).save("./content/"+filename)
    else:
        with_image(init_image, prompt).save("./content/"+filename)
    return send_file("./content/"+filename, mimetype="img/png")


@imageToImage_routes.route('/surprise', methods=['GET'])
def image_to_image_surprise():
    filename = f'image_{uuid.uuid4().hex}.png'
    without_init_image(surpriseCharacter(), surpriseCharacter()).save(
        "./content/"+filename)
    return send_file("./content/"+filename, mimetype='image/png')
