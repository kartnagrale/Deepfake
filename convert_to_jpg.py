from PIL import Image
import os

def convert_to_jpg(folder):
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        try:
            img = Image.open(path).convert("RGB")
            new_path = os.path.join(folder, filename.split('.')[0] + ".jpg")
            img.save(new_path, "JPEG")
            print(f"✅ Converted {filename} to {new_path}")
        except Exception as e:
            print(f"⚠️ Failed to convert {filename}: {e}")

convert_to_jpg("real")
convert_to_jpg("fake")
