from flask import Flask, request, render_template, jsonify, send_file
import os
import torch
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from meta_learner import MetaLearner, train_model, predict
from utils import save_model, load_model

app = Flask(__name__)
UPLOAD_FOLDER = "uploads/"
MODEL_FOLDER = "models/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Initialize Feature Extractor
extractor = FeatureExtractor()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    real_dir = request.form.get("real_dir")
    fake_dir = request.form.get("fake_dir")
    
    if not real_dir or not fake_dir:
        return jsonify({"error": "Please provide paths for real and fake images."})

    X, y = [], []

    # Extract features from real and fake images
    for label, dir_path in [(0, real_dir), (1, fake_dir)]:
        if not os.path.exists(dir_path) or not os.listdir(dir_path):
            return jsonify({"error": f"Directory {dir_path} is empty or does not exist."})
        
        for filename in os.listdir(dir_path):
            path = os.path.join(dir_path, filename)
            try:
                img = Image.open(path).convert("RGB")  # Ensure RGB mode
                vec = extractor.extract(img)
                X.append(vec)
                y.append(label)
            except Exception as e:
                print(f"Error processing {path}: {e}")

    if len(X) < 4:
        return jsonify({"error": "Not enough data. Add more real/fake images."})

    X = np.array(X)
    y = np.array(y)

    # Train Model
    model = MetaLearner()
    model = train_model(model, X, y, epochs=30, lr=0.001)
    
    # Save trained model
    model_path = os.path.join(MODEL_FOLDER, "deepfake_model.pth")
    save_model(model, model_path)

    return jsonify({"message": "Model trained successfully!", "model_path": model_path})

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No image uploaded"})

    file = request.files['file']
    model_file = request.files.get('model')  # Get model file if provided

    if file.filename == "":
        return jsonify({"error": "No selected image file"})

    # Save uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # If a model is uploaded, save and use it; otherwise, use the default model
    if model_file and model_file.filename != "":
        model_path = os.path.join(MODEL_FOLDER, "uploaded_model.pth")
        model_file.save(model_path)
    else:
        model_path = os.path.join(MODEL_FOLDER, "deepfake_model.pth")
        if not os.path.exists(model_path):
            return jsonify({"error": "No trained model found. Please train a model or upload one."})

    # Load the trained or uploaded model
    model = load_model(MetaLearner, model_path)

    try:
        img = Image.open(filepath).convert("RGB")  # Ensure it's in RGB format
        vec = extractor.extract(img).reshape(1, -1)
        pred = predict(model, vec)[0]
        result = "REAL" if pred == 0 else "FAKE"
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": f"Error processing image: {str(e)}"})

if __name__ == "__main__":
    app.run(debug=True)