import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

class FeatureExtractor:
    def __init__(self):
        # Load Pretrained ResNet Model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Remove final classification layer
        self.model.eval()

        # Define Image Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img):
        img = self.transform(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            features = self.model(img).flatten().numpy()  # Extract features and flatten
        return features
