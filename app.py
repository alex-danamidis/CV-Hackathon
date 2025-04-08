from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__)

# Define the path to the model directory
MODEL_PATH = os.path.join(os.getcwd(), 'static')

@app.route('/')
def index():
    # Render the index.html page
    return render_template('index.html')

@app.route('/static/<path:path>')
def send_static(path):
    # Serve static files (e.g., JS, model) from the 'static' folder
    return send_from_directory(MODEL_PATH, path)

if __name__ == '__main__':
    app.run(debug=True)
