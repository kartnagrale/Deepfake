from flask import Flask, request, render_template, jsonify, send_file
import os
import torch
import numpy as np
from PIL import Image
from feature_extractor import FeatureExtractor
from meta_learner import MetaLearner, train_model, predict
from utils import save_model, load_model
from flask import send_file
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    real_files = request.files.getlist("real_images")
    fake_files = request.files.getlist("fake_images")
    
    if not real_files or not fake_files:
        return jsonify({"error": "Please upload both real and fake images."})

    X, y = [], []
    for label, files in [(0, real_files), (1, fake_files)]:
        for file in files:
            img = Image.open(file).convert("RGB")
            vec = extractor.extract(img)
            X.append(vec)
            y.append(label)

    if len(X) < 4:
        return jsonify({"error": "Not enough data to train the model."})

    X = np.array(X)
    y = np.array(y)
    
    # Split data into train and test sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Model
    model = MetaLearner()
    model = train_model(model, X, y, epochs=30, lr=0.001)

    # Save trained model
    model_path = os.path.join(MODEL_FOLDER, "deepfake_model.pth")
    save_model(model, model_path)
    
    # Predict on test data
    y_pred = predict(model, X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    return jsonify({
        "message": "Model trained successfully!",
        "accuracy": round(accuracy * 100, 2),
        "precision": round(precision * 100, 2),
        "recall": round(recall * 100, 2),
        "f1_score": round(f1 * 100, 2),
        "model_download_url": "models/deepfake_model.pth"
    })


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files['file']
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    model_path = os.path.join(MODEL_FOLDER, "deepfake_model.pth")
    if not os.path.exists(model_path):
        return jsonify({"error": "No trained model found."})

    model = load_model(MetaLearner, model_path)

    img = Image.open(filepath).convert("RGB")
    vec = extractor.extract(img).reshape(1, -1)
    pred = predict(model, vec)[0]
    result = "REAL" if pred == 0 else "FAKE"

    return jsonify({"result": result})

@app.route('/download_model')
def download_model():
    model_path = "models/deepfake_model.pth"  # Ensure correct path
    if os.path.exists(model_path):
        return send_file(model_path, as_attachment=True)
    else:
        return jsonify({"error": "Model file not found"}), 404

if __name__ == "__main__":
    app.run(debug=True)