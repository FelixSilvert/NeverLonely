
from flask import Flask
# from routes.auth import auth_routes
from routes.image_to_image_routes import imageToImage_routes
from routes.text_to_image_routes import textToImage_routes
from routes.image_to_image_finetuning_routes import image_to_image_finetuning_routes
# from routes.InpaintingFinetuning import inpaintingFinetuning_routes
from routes.inpainting_routes import inpainting_routes
from routes.generate_manualy_mask_routes import generateManualyMask_routes


app = Flask(__name__)

# # Register blueprints
# app.register_blueprint(auth_routes, url_prefix='/auth')
app.register_blueprint(imageToImage_routes, url_prefix='/imageToImage')
app.register_blueprint(textToImage_routes, url_prefix='/textToImage')
app.register_blueprint(image_to_image_finetuning_routes,
                       url_prefix='/imageToImageFinetuning')
# app.register_blueprint(inpaintingFinetuning_routes,
#                        url_prefix='/inpaintingFinetuning')
app.register_blueprint(inpainting_routes, url_prefix='/inpainting')
app.register_blueprint(generateManualyMask_routes,
                       url_prefix='/generateManualyMask')


if __name__ == "__main__":
    app.run()
