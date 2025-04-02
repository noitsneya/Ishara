import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class ISLLightweightModel:
    def __init__(self, csv_path=None):
        self.find_csv_path(csv_path)
        self.model = None
        self.label_encoder = LabelEncoder()

    def find_csv_path(self, csv_path):
        if csv_path and os.path.exists(csv_path):
            self.csv_path = csv_path
            return
        
        possible_paths = [
            'datasets/isl_data.csv', 
            './datasets/isl_data.csv', 
            '../datasets/isl_data.csv', 
            './isl_data.csv'
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                self.csv_path = path
                print(f"Found data at: {path}")
                return
        
        raise FileNotFoundError("Could not find the ISL dataset CSV file")

    def load_data(self):
        print("Loading data...")
        data = pd.read_csv(self.csv_path)
        print(f"Data loaded successfully with {len(data)} samples")

        # Prepare features and labels
        X = data.iloc[:, 1:].values  # All columns except the first (label)
        y = data.iloc[:, 0].values   # First column contains labels

        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(self.label_encoder.classes_)
        print(f"Number of classes: {self.num_classes}")

        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.X_train.shape[1],)),
            tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        self.model = model
        model.summary()

    def train_model(self, epochs=50, batch_size=32):
        print("Training model...")
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10, restore_best_weights=True
        )
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def evaluate_model(self):
        print("Evaluating model...")
        test_loss, test_acc = self.model.evaluate(self.X_test, self.y_test)
        print(f'Test accuracy: {test_acc:.4f}')

    def plot_training_history(self, history):
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

    def save_model(self):
        # Save TensorFlow Lite model
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        tflite_model = converter.convert()

        os.makedirs('models', exist_ok=True)
        with open('models/isl_alphabet_model.tflite', 'wb') as f:
            f.write(tflite_model)

        # Save label mapping
        import json
        label_mapping = {int(i): label for i, label in enumerate(self.label_encoder.classes_)}
        with open('models/label_mapping.json', 'w') as f:
            json.dump(label_mapping, f)

        print("Model saved successfully!")
        print("TFLite model: models/isl_alphabet_model.tflite")
        print("Label mapping: models/label_mapping.json")

if __name__ == "__main__":
    model = ISLLightweightModel()
    model.load_data()
    model.create_model()
    history = model.train_model()
    model.evaluate_model()
    model.plot_training_history(history)
    model.save_model()
