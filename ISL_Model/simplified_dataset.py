# save as: simplified_dataset.py

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ISLWordDataset(Dataset):
    def __init__(self, csv_file):
        """
        Args:
            csv_file (string): Path to the word details CSV file
        """
        self.word_details = pd.read_csv(csv_file)
        self.words = self.word_details['Word'].unique().tolist()
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}
        
    def __len__(self):
        return len(self.word_details)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get the word at this index
        word = self.word_details.iloc[idx]['Word']
        
        # Convert word to class index
        label = self.word_to_idx[word]
        
        # Create a simple feature vector (one-hot encoding of the word)
        feature = torch.zeros(len(self.words))
        feature[label] = 1.0
        
        return feature, label
    
    def get_word_from_label(self, label):
        """Convert a label back to its word"""
        return self.words[label]
    
    def num_classes(self):
        """Return the number of unique words"""
        return len(self.words)

# For testing the dataset
if __name__ == "__main__":
    # Replace with the actual path to your CSV file
    csv_path = r'C:\Users\dell\OneDrive\Documents\Desktop\Ishara\Ishara\ISL_Model\datasets\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.csv'
    
    # Create an instance of the dataset
    dataset = ISLWordDataset(csv_path)
    
    # Print some information about the dataset
    print(f"Total number of samples: {len(dataset)}")
    print(f"Number of unique words: {dataset.num_classes()}")
    print(f"First 10 words: {dataset.words[:10]}")
    
    # Get and print a sample
    feature, label = dataset[0]
    print(f"\nSample feature shape: {feature.shape}")
    print(f"Sample label: {label}")
    print(f"Corresponding word: {dataset.get_word_from_label(label)}")
