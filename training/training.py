from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
import pandas as pd
from datetime import datetime
import numpy as np
from custom_transformers import MediaPipePreprocessor, HandSizeNormalizer



# Modified training.py with normalization
df = pd.read_csv('Data_Collection/datasets/isl_data.csv')

# Separate features (X) and labels (y)
X = df.drop(columns=['label'])  # All columns except 'label'
y = df['label']

print("Training run start")
training_start = datetime.now()

# Create preprocessing and model pipeline
pipeline = Pipeline([
    ('mediapipe_preprocessing', MediaPipePreprocessor(flip_horizontal=False, stabilize=True)),
    ('hand_size_normalizer', HandSizeNormalizer()),
    ('classifier', RandomForestClassifier(n_estimators=100))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train pipeline
pipeline.fit(X_train, y_train)

training_end = datetime.now()
train_time = training_end - training_start  # Fixed the time calculation
print(f"Training run ended: {train_time} seconds")

# Make predictions
y_pred = pipeline.predict(X_test)

# Evaluate with multiple metrics
accuracy = pipeline.score(X_test, y_test)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Print metrics
print(f"Accuracy: {accuracy:.7f}")
print(f"Precision: {precision:.7f}")
print(f"Recall: {recall:.7f}")
print(f"F1 Score: {f1:.7f}")
print("Confusion Matrix:")
print(cm)

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the entire pipeline as the model
current_time = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
model_filename = f'models/hand_gesture_pipeline_{current_time}.joblib'
joblib.dump(pipeline, model_filename)
print(f"Model saved to {model_filename}")
