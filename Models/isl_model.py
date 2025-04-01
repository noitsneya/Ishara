# isl_alphabet_model.py
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# 1. Load and prepare data
print("Loading data...")
try:
    # Try different possible paths to find the CSV file
    possible_paths = ['datasets/isl_data.csv', './datasets/isl_data.csv', '../datasets/isl_data.csv', './isl_data.csv']
    csv_path = None
    
    for path in possible_paths:
        if os.path.exists(path):
            csv_path = path
            print(f"Found data at: {path}")
            break
    
    if csv_path is None:
        raise FileNotFoundError("Could not find the ISL dataset CSV file")
    
    data = pd.read_csv(csv_path)
    print(f"Data loaded successfully with {len(data)} samples")
    print(f"Available letters: {data['label'].unique()}")
    
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

# 2. Prepare features and labels
X = data.iloc[:, 1:].values  # All columns except the first (label)
y = data.iloc[:, 0].values   # First column contains labels

# 3. Encode labels (convert A, B, C to numbers)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# 4. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Build a lightweight model optimized for mobile
def create_lightweight_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        # Input layer
        tf.keras.layers.Input(shape=(input_shape,)),
        
        # First dense layer with L2 regularization to prevent overfitting
        tf.keras.layers.Dense(64, activation='relu', 
                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        
        # Second dense layer
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        
        # Output layer
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile model with Adam optimizer
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Create model
print("Creating model...")
model = create_lightweight_model(X_train.shape[1], num_classes)
model.summary()

# 6. Train model with early stopping
print("Training model...")
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.0001
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr],
    verbose=1
)

# 7. Evaluate model
print("Evaluating model...")
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# 8. Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('models/training_history.png')
print("Training history saved to models/training_history.png")

# 9. Convert to TensorFlow Lite
print("Converting to TensorFlow Lite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Apply optimizations for mobile
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

# 10. Save TFLite model
with open('models/isl_alphabet_model.tflite', 'wb') as f:
    f.write(tflite_model)

# 11. Save label mapping
import json
label_mapping = {int(i): label for i, label in enumerate(label_encoder.classes_)}
with open('models/label_mapping.json', 'w') as f:
    json.dump(label_mapping, f)

# 12. Save Keras model for reference
model.save('models/isl_alphabet_model.h5')

print("Model training complete!")
print(f"TFLite model saved to: models/isl_alphabet_model.tflite")
print(f"Label mapping saved to: models/label_mapping.json")
print(f"Keras model saved to: models/isl_alphabet_model.h5")
