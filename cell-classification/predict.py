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

# Define the same transforms used during training
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224
    transforms.ToTensor(),          # Convert PIL image to PyTorch tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize as per ImageNet
                         std=[0.229, 0.224, 0.225])
])

# Prediction function
def predict(image_path, model):
    # Open the image file
    try:
        image = Image.open(image_path)
        # Handle different image modes
        if image.mode.startswith('I;16'):
            # Convert 16-bit image to 8-bit
            numpy_image = np.array(image, dtype=np.uint16)
            numpy_image = (numpy_image / 256).astype('uint8')
            image = Image.fromarray(numpy_image, mode='L')
            image = image.convert('RGB')
        else:
            image = image.convert('RGB')
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    # Apply transforms
    input_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Model inference
    with torch.no_grad():
        output = model(input_tensor)
        output = output.view(-1)
        prob = output.item()
        predicted_label = int(prob >= 0.5)  # Threshold at 0.5

    return predicted_label, prob

if __name__ == '__main__':
    # Specify the path to your image
    image_path = 'data/training/245.tif'  # Replace with your image path

    # Load the model
    model = load_model('best_model.pth')  # Adjust the path if needed

    # Make the prediction
    result = predict(image_path, model)

    if result is not None:
        predicted_label, probability = result
        # Output the prediction
        label_name = 'Homogeneous' if predicted_label == 1 else 'Heterogeneous'
        print(f"Predicted Label: {label_name} ({predicted_label})")
        print(f"Probability: {probability:.4f}")
    else:
        print("Prediction could not be made due to an error.")
