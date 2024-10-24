import uuid
from utils.text_to_image_utils import text_to_image
from flask import Blueprint, request, send_file
from random import randint


# Create a Blueprint for imageToImage routes
textToImage_routes = Blueprint('textToImage_routes', __name__)

# Define the routes for the Blueprint


def get_param(param_name, default_value=None):
    value = request.args.get(param_name)
    if not value:
        return default_value
    return value


@textToImage_routes.route('/', methods=['GET'])
def textToImage():
    artist_style = get_param('artist_style', "")
    negative_prompt_value = get_param('negative_prompt_value', "")
    num_inference_steps_value = get_param('num_inference_steps_value', "")
    guidance_scale_value = get_param('guidance_scale_value', "")
    avatar_or_illustration = get_param('avatar_or_illustration', "Avatar")
    seed = get_param('seed', randint(0, 1000000))
    lighting = get_param('lighting', "")
    environment = get_param('environment', "")
    color_scheme = get_param('color_scheme', "")
    point_of_view = get_param('point_of_view', "")
    background = get_param('background', "")
    art_style = get_param('art_style', "")
    person_group_detail = get_param('person_group_detail', "")

    # Generate a unique filename using UUID
    filename = f'image_{uuid.uuid4().hex}.png'

    image, prompt = text_to_image(
        artist_style=artist_style,
        negative_prompt_value=negative_prompt_value,
        num_inference_steps_value=num_inference_steps_value,
        guidance_scale_value=guidance_scale_value,
        avatar_or_illustration=avatar_or_illustration,
        seed=seed,
        lighting=lighting,
        environment=environment,
        color_scheme=color_scheme,
        point_of_view=point_of_view,
        background=background,
        art_style=art_style,
        person_group_detail=person_group_detail
    )

    image.save("./content/"+filename)

    return send_file("./content/"+filename, mimetype='image/png')
