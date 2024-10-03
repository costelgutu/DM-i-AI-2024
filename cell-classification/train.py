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
csv_file = 'data/training.csv'  # CSV file with image_id and is_homogeneous
IMG_SIZE = (224, 224)
BATCH_SIZE = 8
df = pd.read_csv(csv_file)
def load_and_preprocess_image(image_path):
    # Load image with PIL and convert to an array
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    # Normalize image pixel values (0-255 -> 0-1)
    img_array = img_array / 255.0
    return img_array

print(df.columns)
df.columns = df.columns.str.strip()

# 3. Create lists of image paths and labels
# Assuming image_id values need to be 3 digits with leading zeros
image_paths = [os.path.join(data_dir, f"{str(image_id).zfill(3)}.tif") for image_id in df['image_id']]
labels = df['is_homogenous'].values
images = np.array([load_and_preprocess_image(image_path) for image_path in image_paths])
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
# 6. Create data generators for augmentation
train_datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()
# 7. Create the data generators
train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE)
val_generator = val_datagen.flow(X_val, y_val, batch_size=BATCH_SIZE)
# 8. Load the pre-trained VGG16 model without the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# 10. Create the model by adding custom layers on top of the pre-trained base model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),  # Add dropout for regularization
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])
# 11. Compile the model
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 12. Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust the number of epochs as needed
    validation_data=val_generator
)
model.save('vgg16_homogeneous_classification.h5')
# 13. Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")
predictions = model.predict(X_val)
predicted_labels = (predictions >= 0.5).astype(int)  # Threshold at 0.5 to get binary labels
from sklearn.metrics import classification_report
print(classification_report(y_val, predicted_labels, target_names=['Heterogeneous', 'Homogeneous']))


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


