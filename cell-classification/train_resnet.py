# Import necessary libraries
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import keras_tuner as kt
import tensorflow_addons as tfa  # For Focal Loss if needed

# Paths
data_dir = 'data/training'  # Folder with .tif images
csv_file = 'data/training.csv'  # CSV file with image_id and is_homogeneous
IMG_SIZE = (224, 224)
BATCH_SIZE = 4

# Load the CSV file
df = pd.read_csv(csv_file)

# Ensure correct column names
df.columns = df.columns.str.strip()
df.rename(columns={'is_homogenous': 'is_homogeneous'}, inplace=True)

# Create lists of image paths and labels
image_paths = [os.path.join(data_dir, f"{str(image_id).zfill(3)}.tif") for image_id in df['image_id']]
labels = df['is_homogeneous'].values

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Load images
images = np.array([load_and_preprocess_image(image_path) for image_path in image_paths])

# Stratify train-test split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, random_state=42, stratify=labels
)

def build_model(hp):
    # Load the pre-trained ResNet50 model without the top layer
    base_model = ResNet50(
        weights='imagenet', include_top=False, input_shape=(224, 224, 3)
    )
    
    # Freeze all layers initially
    for layer in base_model.layers:
        layer.trainable = False
    
    # Option to unfreeze last n layers
    unfreeze_layers = hp.Int('unfreeze_layers', min_value=0, max_value=10, step=1)
    if unfreeze_layers > 0:
        for layer in base_model.layers[-unfreeze_layers:]:
            layer.trainable = True
    
    # Add custom layers on top of the base model
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    
    # Tune the number of units and dropout rate
    units = hp.Int('units', min_value=64, max_value=256, step=64)
    x = Dense(units, activation='relu')(x)
    
    dropout_rate = hp.Float('dropout_rate', min_value=0.3, max_value=0.7, step=0.1)
    x = Dropout(dropout_rate)(x)
    
    # Second Dense layer
    units2 = hp.Int('units2', min_value=32, max_value=128, step=32)
    x = Dense(units2, activation='relu')(x)
    
    # Output layer
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create the model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Tune the learning rate
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-3, sampling='LOG')
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Subclass RandomSearch tuner
class MyTuner(kt.tuners.RandomSearch):
    def run_trial(self, trial, X_train, y_train, X_val, y_val):
        hp = trial.hyperparameters
        
        # Create data augmentation parameters using hp
        train_datagen = ImageDataGenerator(
            rotation_range=hp.Int('rotation_range', 0, 40, step=10),
            zoom_range=hp.Float('zoom_range', 0.0, 0.3, step=0.1),
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
        
        # No augmentation for validation data
        val_datagen = ImageDataGenerator()
        
        # Create data generators
        train_generator = train_datagen.flow(
            X_train, y_train, batch_size=BATCH_SIZE
        )
        val_generator = val_datagen.flow(
            X_val, y_val, batch_size=BATCH_SIZE
        )
        
        # Build model
        model = self.hypermodel.build(hp)
        
        # Get class weights
        class_weight_0 = hp.Float('class_weight_0', 0.5, 2.0, step=0.5)
        class_weight_1 = hp.Float('class_weight_1', 1.0, 5.0, step=1.0)
        class_weights = {0: class_weight_0, 1: class_weight_1}
        
        # Add callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
        ]
        
        # Fit the model
        history = model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=10,
            callbacks=callbacks,
            class_weight=class_weights,
            verbose=0  # Suppress training output for clarity
        )
        
        # Report the final accuracy to the tuner
        val_loss, val_accuracy = model.evaluate(val_generator, verbose=0)
        self.oracle.update_trial(
            trial.trial_id, {'val_accuracy': val_accuracy}
        )

# Initialize the tuner
tuner = MyTuner(
    hypermodel=build_model,
    objective='val_accuracy',
    max_trials=20,
    executions_per_trial=1,
    directory='hyperparam_tuning',
    project_name='resnet50_tuning'
)

# Run the hyperparameter search
tuner.search(
    X_train, y_train, X_val, y_val
)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Print the best hyperparameters
print(f"""
The hyperparameter search is complete. The optimal parameters are:
- Units: {best_hps.get('units')}
- Units2: {best_hps.get('units2')}
- Dropout Rate: {best_hps.get('dropout_rate')}
- Learning Rate: {best_hps.get('learning_rate')}
- Unfreeze Layers: {best_hps.get('unfreeze_layers')}
- Class Weight 0: {best_hps.get('class_weight_0')}
- Class Weight 1: {best_hps.get('class_weight_1')}
- Rotation Range: {best_hps.get('rotation_range')}
- Zoom Range: {best_hps.get('zoom_range')}
""")

# Build the best model
best_model = tuner.hypermodel.build(best_hps)

# Update data augmentation parameters with best hyperparameters
train_datagen = ImageDataGenerator(
    rotation_range=best_hps.get('rotation_range'),
    zoom_range=best_hps.get('zoom_range'),
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator()

train_generator = train_datagen.flow(
    X_train, y_train, batch_size=BATCH_SIZE
)
val_generator = val_datagen.flow(
    X_val, y_val, batch_size=BATCH_SIZE
)

# Compute class weights
class_weights = {
    0: best_hps.get('class_weight_0'),
    1: best_hps.get('class_weight_1')
}

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

# Train the best model
history = best_model.fit(
    train_generator,
    epochs=30,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)

# Evaluate the model
val_loss, val_acc = best_model.evaluate(val_generator)
print(f"Validation Accuracy: {val_acc*100:.2f}%")

# Generate predictions
predictions = best_model.predict(X_val)
predicted_labels = (predictions >= 0.5).astype(int)

# Classification report
from sklearn.metrics import classification_report
print(classification_report(y_val, predicted_labels, target_names=['Heterogeneous', 'Homogeneous']))

# Custom score calculation
n_0 = np.sum(y_val == 0)
n_1 = np.sum(y_val == 1)
a_0 = np.sum((y_val == 0) & (predicted_labels.flatten() == 0))
a_1 = np.sum((y_val == 1) & (predicted_labels.flatten() == 1))

if n_0 == 0 or n_1 == 0:
    score = 0
else:
    score = (a_0 * a_1) / (n_0 * n_1)

print(f'Score: {score}')
