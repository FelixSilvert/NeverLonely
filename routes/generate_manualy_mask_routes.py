from flask import request, Blueprint, render_template, jsonify
import base64
import io
from PIL import Image


generateManualyMask_routes = Blueprint('generateManualyMask_routes', __name__)


@generateManualyMask_routes.route('/')
def index():
    return render_template('index.html')

# Handle mask upload and save the mask image


@generateManualyMask_routes.route('/upload_mask', methods=['POST'])
def upload_mask():
    data = request.json['mask']
    # Remove the 'data:image/png;base64,' part
    mask_data = data.split(',')[1]

    # Decode the image
    mask_image = base64.b64decode(mask_data)

    # Convert the image into a PIL object
    image = Image.open(io.BytesIO(mask_image)).convert("RGBA")

    # Create a new image with a black background
    black_bg = Image.new("RGBA", image.size, "black")

    # Paste the mask image onto the black background
    black_bg.paste(image, (0, 0), image)

    # change the red color to white
    data = black_bg.getdata()
    newData = []
    for item in data:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append(item)
        else:
            newData.append((255, 255, 255, 255))
    # update image data
    black_bg.putdata(newData)

    # Save the image as a PNG file
    black_bg.save('mask_image.png')

    # Replace 'some_other_route' with your actual route name
    other_url = url_for(
        'http://127.0.0.1/inpainting/manualy?mask_image=mask_image.png')
    return redirect(other_url)
