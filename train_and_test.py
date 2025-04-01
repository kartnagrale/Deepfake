import os
from feature_extractor import FeatureExtractor
from meta_learner import MetaLearner, train_model, predict
import numpy as np
from PIL import Image
import torch

# Step 1: Prepare data
def load_data(real_dir, fake_dir, extractor):
    X = []
    y = []

    for filename in os.listdir(real_dir):
        path = os.path.join(real_dir, filename)
        print(f"Trying to load: {path}")  # ✅ Print for debugging
        try:
            vec = extractor.extract(Image.open(path))
            X.append(vec)
            y.append(0)  # Real label
        except Exception as e:
            print(f"⚠️ Error processing real image {path}: {e}")  # Show exact error

    for filename in os.listdir(fake_dir):
        path = os.path.join(fake_dir, filename)
        try:
            vec = extractor.extract(Image.open(path))
            X.append(vec)
            y.append(1)  # Fake label
        except:
            print(f"Error processing fake image: {path}")

    return np.array(X), np.array(y)

# Step 2: Train
def main():
    extractor = FeatureExtractor()
    real_dir = "real"
    fake_dir = "fake"

    print("🔄 Extracting features...")
    X, y = load_data(real_dir, fake_dir, extractor)

    if len(X) < 4:
        print("❌ Not enough data. Add more real/fake images.")
        return

    model = MetaLearner()
    model = train_model(model, X, y, epochs=30, lr=0.001)

    # Save the model if you want later
    torch.save(model.state_dict(), "meta_model.pth")

    # Step 3: Predict on a test image
    test_img_path = "test3.jpg"  # Replace with any test image path
    vec = extractor.extract(Image.open(test_img_path)).reshape(1, -1)
    pred = predict(model, vec)[0]

    if pred == 0:
        print("✅ The image is REAL.")
    else:
        print("❌ The image is FAKE.")

if __name__ == "__main__":
    main()
