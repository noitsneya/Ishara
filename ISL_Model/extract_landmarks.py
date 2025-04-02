import os
import pandas as pd
import numpy as np
import sign_language_translator as slt
from tqdm import tqdm

def extract_landmarks(word_details_csv, output_file):
    """
    Extract landmarks for each word in the word details CSV.

    Args:
        word_details_csv (str): Path to the ISL-CSLTR word details CSV file.
        output_file (str): Path to save extracted landmarks as a .npy file.
    """
    # Load word details
    word_details = pd.read_csv(word_details_csv)

    # Debug: Print column names and check for missing values
    print("Word Details Columns:", word_details.columns.tolist())
    print("\nMissing values in word details:")
    print(word_details.isna().sum())

    # Ensure columns are correctly named and strip whitespace if necessary
    word_details.columns = word_details.columns.str.strip()

    # Drop rows with missing frame paths or invalid words
    word_details = word_details.dropna(subset=['Word', 'Frames path'])

    # Ensure valid paths exist for frames
    word_details = word_details[word_details['Frames path'].apply(os.path.exists)]

    # Initialize MediaPipeLandmarksModel
    embedding_model = slt.models.MediaPipeLandmarksModel()

    # Prepare storage for landmarks
    landmarks_data = []

    # Iterate through all rows in the word details CSV
    for _, row in tqdm(word_details.iterrows(), total=len(word_details)):
        word = row['Word']
        frame_path = row['Frames path']

        if not os.path.exists(frame_path):
            print(f"Frame not found: {frame_path}")
            continue

        try:
            # Load frame and extract landmarks
            frame = slt.utils.load_image(frame_path)
            landmarks = embedding_model.embed([frame])
            landmarks_data.append((word, landmarks))
        except Exception as e:
            print(f"Error processing {frame_path}: {e}")

    # Save extracted landmarks
    np.save(output_file, landmarks_data)
    print(f"Landmarks saved to {output_file}")

if __name__ == "__main__":
    # Paths (update these with actual paths)
    word_details_csv = r"C:\Users\dell\OneDrive\Documents\Desktop\Ishara\Ishara\ISL_Model\datasets\ISL_CSLRT_Corpus\corpus_csv_files\ISL_CSLRT_Corpus_word_details.csv"
    output_file = r"C:\Users\dell\OneDrive\Documents\Desktop\Ishara\landmarks.npy"

    extract_landmarks(word_details_csv, output_file)
