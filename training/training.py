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

# Add the hand size normalization function as a transformer
class HandSizeNormalizer(BaseEstimator, TransformerMixin):
    def __init__(self, landmark_count=21):
        self.landmark_count = landmark_count
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        normalized_data = []
        num_samples = X.shape[0]
        
        for i in range(num_samples):
            # Reshape to (21 landmarks, 3 coordinates)
            sample = X[i].reshape(self.landmark_count, 3)
            
            # Get wrist position (first landmark in MediaPipe hand tracking)
            wrist = sample[0]
            
            # Center around wrist
            centered = sample - wrist
            
            # Find maximum distance from wrist to any landmark
            distances = np.linalg.norm(centered, axis=1)
            max_distance = np.max(distances)
            
            # Avoid division by zero
            if max_distance == 0:
                max_distance = 1.0
                
            # Normalize by this distance
            normalized_sample = centered / max_distance
            
            # Flatten and add to result
            normalized_data.append(normalized_sample.flatten())
        
        return np.array(normalized_data)

# MediaPipe specific preprocessing
class MediaPipePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, flip_horizontal=False, stabilize=True):
        self.flip_horizontal = flip_horizontal
        self.stabilize = stabilize
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        processed_data = []
        num_samples = X.shape[0]
        
        for i in range(num_samples):
            # Reshape to (21 landmarks, 3 coordinates)
            sample = X[i].reshape(21, 3)
            
            # Optional: Flip horizontally (for left/right hand consistency)
            if self.flip_horizontal:
                sample[:, 0] = -sample[:, 0]  # Flip x-coordinates
                
            # Optional: Stabilization (reduce jitter)
            if self.stabilize:
                # Simple stabilization: set very small movements to zero
                threshold = 0.01
                sample[np.abs(sample) < threshold] = 0
                
            processed_data.append(sample.flatten())
            
        return np.array(processed_data)

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
