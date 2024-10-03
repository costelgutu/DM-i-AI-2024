import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib

# Check if TensorFlow is using a GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow is using the following GPU(s):")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU detected for TensorFlow.")

# Paths
data_dir = 'data/training'  # Folder with .tif images
csv_file = 'data/training.csv'  # CSV file with image_id and is_homogenous
IMG_SIZE = (224, 224)
BATCH_SIZE = 8

# Load the CSV file
df = pd.read_csv(csv_file)
df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces in column names

print(df.columns)

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    # Load image with PIL and convert to an array
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    # Normalize image pixel values (0-255 -> 0-1)
    img_array = img_array / 255.0
    return img_array

# Create lists of image paths and labels
# Assuming image_id values need to be 3 digits with leading zeros
image_paths = [os.path.join(data_dir, f"{str(image_id).zfill(3)}.tif") for image_id in df['image_id']]
labels = df['is_homogenous'].values

# Load images
images = np.array([load_and_preprocess_image(image_path) for image_path in image_paths])

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Create data generators for augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

# Create the data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)

# Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Define the custom F1Score metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.tp = self.add_weight(name='true_positives', initializer='zeros')
        self.fp = self.add_weight(name='false_positives', initializer='zeros')
        self.fn = self.add_weight(name='false_negatives', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions and labels to binary tensors
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)

        # Update true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true * y_pred)
        fp = tf.reduce_sum((1 - y_true) * y_pred)
        fn = tf.reduce_sum(y_true * (1 - y_pred))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + K.epsilon())
        recall = self.tp / (self.tp + self.fn + K.epsilon())
        f1 = 2 * precision * recall / (precision + recall + K.epsilon())
        return f1

    def reset_states(self):
        self.tp.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

# Create the model by adding custom layers on top of the pre-trained base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model with the custom F1Score metric
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=[F1Score()])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator
)

# Save the trained model
model.save('vgg16_homogeneous_classification.h5')

# Evaluate the model on the validation set
val_loss, val_f1 = model.evaluate(val_generator)
print(f"Validation F1-Score: {val_f1*100:.2f}%")

# Generate predictions and classification report
predictions = model.predict(X_val)
predicted_labels = (predictions >= 0.5).astype(int)  # Threshold at 0.5 to get binary labels

from sklearn.metrics import classification_report
print(classification_report(y_val, predicted_labels, target_names=['Heterogeneous', 'Homogeneous']))

# Calculate the custom score as before
# Assuming y_val contains the true labels and predicted_labels contains the predicted labels

# Step 1: Calculate n_0 and n_1
n_0 = np.sum(y_val == 0)  # Number of true heterogeneous cells
n_1 = np.sum(y_val == 1)  # Number of true homogeneous cells

# Step 2: Calculate a_0 and a_1
a_0 = np.sum((y_val == 0) & (predicted_labels == 0))  # Correctly predicted as heterogeneous
a_1 = np.sum((y_val == 1) & (predicted_labels == 1))  # Correctly predicted as homogeneous

# Step 3: Calculate the score
if n_0 == 0 or n_1 == 0:
    score = 0  # Handle edge cases where there are no samples of a class
else:
    score = (a_0 * a_1) / (n_0 * n_1)

print(f'Score: {score}')
