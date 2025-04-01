"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


# Load the CSV file
df = pd.read_csv('datasets/isl_data.csv')

# Separate features and labels
X = df.iloc[:, 1:].values  # All columns except the first (label)
y = df.iloc[:, 0].values   # First column contains labels

# Check for missing values
print(f"Missing values: {df.isnull().sum().sum()}")

# Fill missing values (if any)
df = df.fillna(method='ffill')  # Forward fill

def normalize_min_max(landmarks):
    #Normalize landmarks to [0,1] range.
    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(landmarks)
    return normalized

X_normalized = normalize_min_max(X)

def normalize_to_wrist(landmarks_array):
    #Normalize landmarks relative to wrist position.
    normalized_data = []
    
    # Reshape to get individual samples
    num_samples = landmarks_array.shape[0]
    for i in range(num_samples):
        # Extract the current sample
        sample = landmarks_array[i].reshape(21, 3)  # 21 landmarks, 3 coordinates each
        
        # Get wrist position (assuming it's the first landmark)
        wrist = sample[0]
        
        # Normalize all landmarks relative to wrist
        normalized_sample = sample - wrist
        
        # Flatten and add to result
        normalized_data.append(normalized_sample.flatten())
    
    return np.array(normalized_data)

X_wrist_normalized = normalize_to_wrist(X)

def normalize_by_hand_size(landmarks_array):
    #Normalize landmarks by the size of the hand.
    normalized_data = []
    
    # Reshape to get individual samples
    num_samples = landmarks_array.shape[0]
    for i in range(num_samples):
        # Extract the current sample
        sample = landmarks_array[i].reshape(21, 3)  # 21 landmarks, 3 coordinates each
        
        # Get wrist position
        wrist = sample[0]
        
        # Center around wrist
        centered = sample - wrist
        
        # Find the maximum distance from wrist to any landmark
        distances = np.linalg.norm(centered, axis=1)
        max_distance = np.max(distances)
        
        # Normalize by this distance
        normalized_sample = centered / max_distance
        
        # Flatten and add to result
        normalized_data.append(normalized_sample.flatten())
    
    return np.array(normalized_data)

X_size_normalized = normalize_by_hand_size(X)

def augment_landmarks(landmarks, num_augmentations=5, noise_level=0.01):
    #Add slight noise to landmarks for data augmentation.
    augmented_data = [landmarks]
    
    for _ in range(num_augmentations):
        # Add random noise
        noise = np.random.normal(0, noise_level, landmarks.shape)
        augmented = landmarks + noise
        augmented_data.append(augmented)
    
    return np.vstack(augmented_data)

# Augment only the training data
X_train, X_test, y_train, y_test = train_test_split(
    X_size_normalized, y, test_size=0.2, random_state=42, stratify=y
)

X_train_augmented = augment_landmarks(X_train)
y_train_augmented = np.repeat(y_train, 6)  # Original + 5 augmentations


# Select the most important features
selector = SelectKBest(f_classif, k=40)  # Select top 40 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Get indices of selected features for future reference
selected_indices = selector.get_support(indices=True)


# Convert labels to numerical format if needed
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Save the preprocessing parameters
import pickle

preprocessing_params = {
    'label_encoder': label_encoder,
    'selected_features': selected_indices if 'selected_indices' in locals() else None
}

with open('models/preprocessing_params.pkl', 'wb') as f:
    pickle.dump(preprocessing_params, f)

def preprocess_landmarks_for_training(csv_path):
    #Complete preprocessing pipeline for landmark data.
    # Load data
    df = pd.read_csv(csv_path)
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values
    
    # Normalize by hand size
    X_normalized = normalize_by_hand_size(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_normalized, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)
    
    # Save preprocessing parameters
    preprocessing_params = {
        'label_encoder': label_encoder
    }
    
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessing_params.pkl', 'wb') as f:
        pickle.dump(preprocessing_params, f)
    
    return X_train, X_test, y_train_encoded, y_test_encoded, label_encoder

    """