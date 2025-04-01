from feature_extractor import FeatureExtractor
from PIL import Image

# Load Feature Extractor
extractor = FeatureExtractor()

# Test on a single image
img_path = "real/r1.jpg"  # Change this to an actual image path

try:
    img = Image.open(img_path)
    vec = extractor.extract(img)
    print("✅ Feature extraction successful!")
    print("Feature Vector Length:", len(vec))
    print("First 10 Features:", vec[:10])
except Exception as e:
    print(f"⚠️ Error extracting features: {e}")
