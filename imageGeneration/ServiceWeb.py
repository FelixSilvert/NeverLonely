from flask import Flask, jsonify

app = Flask(__name__)

# Exemple d'une route GET.
@app.route('/api/connecter', methods=['GET'])
def connecter():
    # Se connecter ici à la base de données.
    # Implanter toutes les opérations sur la manipulation des données.
    return jsonify(message='La connexion est bien établie.')

if __name__ == '__main__':
    # Lancement de l'application.
    app.run(debug=True)
