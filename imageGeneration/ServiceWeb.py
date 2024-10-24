from flask import Flask, request, jsonify, send_file
import io
from PIL import Image
import random

app = Flask(__name__)


# Simuler la génération d'une image via une IA (pour cet exemple)
def generate_image_from_query(query):
    # Ici, on peut avoir un modèle IA qui génère une image. Pour l'exemple, on va juste générer une image aléatoire.
    print(query)
    image = Image.new('RGB', (256, 256), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)
    return img_byte_arr


@app.route('/generate_image', methods=['GET'])
def generate_image():
    query = request.args.get('query')

    if not query:
        return jsonify({'error': 'Query parameter is missing'}), 400

    # Appeler une fonction de génération d'image avec la query (simulée ici)
    img_data = generate_image_from_query(query)

    # Retourner l'image sous forme de fichier
    return send_file(img_data, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
