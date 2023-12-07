import io
import os
import sys
from PIL import Image
import torch
from flask import Flask, request, redirect, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = "a_secret_key"

RESULT_FOLDER = os.path.join('static', 'prediction_images')
app.config['RESULT_FOLDER'] = RESULT_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def find_model_name():
    for f in os.listdir():
        if f.endswith(".pt"):
            return f
    print("Fichier introuvable")
    return None
    
model_name = find_model_name()
if model_name:
    model = torch.hub.load("WongKinYiu/yolov7", 'custom', model_name, verbose=False)
    #model.eval()
else:
    sys.exit("Quitter: le fichier modèle est requis pour exécuter l'application.")

def get_prediction(model, img_bytes):
    img = Image.open(io.BytesIO(img_bytes))
    imgs = [img]
    results = model(imgs, size=320)
    return results

def save_file(file):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['RESULT_FOLDER'], filename)
    file.save(filepath)
    return filepath

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('file')
        if 'file' not in request.files or file.filename == '' or not allowed_file(file.filename):
            return redirect(request.url)
        
        filepath = save_file(file)
        img_bytes = open(filepath, 'rb').read()
        results = get_prediction(model, img_bytes)
        results.save(save_dir=RESULT_FOLDER)
        return render_template('result.html', result_image=filepath, model_name=model_name)
    return render_template('index.html')

if __name__ == "__main__":
    app.run()
