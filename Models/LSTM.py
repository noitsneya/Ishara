import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import to_categorical

# Load the CSV file
def load_data(file_path):
    # Load data assuming the first column is 'label'
    data = pd.read_csv(file_path)
    X = data.iloc[:, 1:].values  # Features (all columns except 'label')
    y = data.iloc[:, 0].values  # Labels (first column)
    return X, y

# Preprocess the data
def preprocess_data(X, y):
    # Reshape X for LSTM input (samples, timesteps, features)
    num_samples = X.shape[0]
    num_timesteps = 2  # Example: split into left and right hand data (adjust as needed)
    num_features = X.shape[1] // num_timesteps
    
    X_reshaped = X.reshape(num_samples, num_timesteps, num_features)
    
    # Encode labels to integers and one-hot encode them
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_one_hot = to_categorical(y_encoded)
    
    return X_reshaped, y_one_hot, label_encoder

# Build the LSTM model
def build_lstm_model(input_shape, num_classes):
    model = Sequential()
    
    # Add LSTM layer(s)
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))  # 64 units in the LSTM layer
    
    # Add Dense output layer with softmax activation for classification
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train and evaluate the model
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    # Train the model
    history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
    
    # Evaluate on test set
    loss, accuracy = model.evaluate(X_test, y_test)
    
    print(f"Test Loss: {loss}")
    print(f"Test Accuracy: {accuracy}")
    
    return history

# Main function to execute the pipeline
def main():
    # Path to your CSV file
    file_path = "C://Users//dell//OneDrive//Documents//Desktop//Ishara//Ishara//Data_Collection//datasets//isl_data.csv"  # Replace with your actual file path
    
    # Load and preprocess data
    X, y = load_data(file_path)
    X_reshaped, y_one_hot, label_encoder = preprocess_data(X, y)
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_one_hot, test_size=0.2, random_state=42)
    
    # Build the LSTM model
    input_shape = (X_reshaped.shape[1], X_reshaped.shape[2])  # (timesteps, features per timestep)
    num_classes = y_one_hot.shape[1]
    
    lstm_model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
    
    # Train and evaluate the model
    train_and_evaluate_model(lstm_model, X_train, y_train, X_test, y_test)

    # Save the trained model in native Keras format (.keras)
    lstm_model.save("lstm_hand_model.keras")
    print("Model saved successfully to 'lstm_hand_model.keras'")
    
    # Save LabelEncoder classes for later use in prediction script
    np.save("label_classes.npy", label_encoder.classes_)
    print("LabelEncoder classes saved to 'label_classes.npy'")
  

# Run the program
if __name__ == "__main__":
    main()
