# model.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture (same as in your training code)
def get_model():
    # Load pre-trained ResNet50 model
    model = models.resnet50(pretrained=False)  # We'll load our own weights
    num_ftrs = model.fc.in_features
    # Define custom classifier
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model

# Load the model
def load_model(model_path='best_model.pth'):
    model = get_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Initialize the model once when the module is imported
model = load_model('best_model.pth')  # Ensure the path is correct

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize as per ImageNet
                         std=[0.229, 0.224, 0.225])
])

def predict(image: np.ndarray) -> int:
    """
    Predicts whether the cell image is homogenous or not.

    Args:
        image (np.ndarray): The input image as a NumPy array.

    Returns:
        int: 1 if homogenous, 0 otherwise.
    """
    try:
        # Convert NumPy array to PIL Image
        if image.ndim == 2:
            # Grayscale to RGB
            image = Image.fromarray(image).convert('RGB')
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = Image.fromarray(image).convert('RGB')
        else:
            image = Image.fromarray(image).convert('RGB')

        # Apply transforms
        input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Model inference
        with torch.no_grad():
            output = model(input_tensor)
            prob = output.item()
            predicted_label = int(prob >= 0.5)  # Threshold at 0.5

        return predicted_label
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return 0  # Default to 0 in case of error
