import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model once when the module is loaded
model = load_model('vgg16_homogeneous_classification.h5')  # Replace with the path to your saved model

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image to match the model's expected input.
    This includes resizing, normalizing, and any other preprocessing steps.
    """
    # Resize the image to match the input size of the model (e.g., 160x160 or 224x224)
    img_size = (160, 160)  # Replace with the size you used for training (160, 160 or 224, 224)
    
    # Assuming image is a numpy array in its original size
    image_resized = tf.image.resize(image, img_size)
    
    # Normalize the image (if required by the model)
    image_normalized = image_resized / 255.0  # Assuming the model was trained on normalized images
    
    # Add a batch dimension (models expect a batch of inputs)
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    return image_batch

### CALL YOUR CUSTOM MODEL VIA THIS FUNCTION ###
def predict(image: np.ndarray) -> int:
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    
    # Make the prediction using the loaded model
    prediction = model.predict(preprocessed_image)
    
    # Since it's a binary classification, the output will be a probability between 0 and 1.
    # We can threshold the prediction at 0.5 to classify as homogenous (1) or heterogenous (0).
    is_homogeneous = 1 if prediction >= 0.5 else 0

    return is_homogeneous
