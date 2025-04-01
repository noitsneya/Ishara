# custom_transformers.py
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd

class MediaPipePreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, flip_horizontal=False, stabilize=True):
        self.flip_horizontal = flip_horizontal
        self.stabilize = stabilize

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        processed_data = []
        for sample in X:
            sample = sample.reshape(21, 3)  # Assuming 21 landmarks with 3 coordinates each
            if self.flip_horizontal:
                sample[:, 0] = -sample[:, 0]  # Flip x-coordinates
            if self.stabilize:
                sample[np.abs(sample) < 0.01] = 0  # Stabilize small movements
            processed_data.append(sample.flatten())
        return np.array(processed_data)


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