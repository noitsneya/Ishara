import cv2
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
import mediapipe as mp

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Function to preprocess hand landmarks for prediction
def preprocess_hand_landmarks(landmarks):
    # Flatten the landmarks into a single array
    flattened_landmarks = np.array(landmarks).flatten()
    # Reshape for LSTM input (samples, timesteps, features)
    num_timesteps = 2  # Example: left and right hand split (adjust as needed)
    num_features = len(flattened_landmarks) // num_timesteps
    return flattened_landmarks.reshape(-1, num_timesteps, num_features)

# Function to make predictions using the trained model
def predict_with_model(model, data, label_encoder):
    predictions = model.predict(data)
    predicted_classes = np.argmax(predictions, axis=1)  # Get class indices from probabilities
    confidence_scores = np.max(predictions, axis=1)  # Get confidence scores for each prediction
    decoded_labels = label_encoder.inverse_transform(predicted_classes)  # Decode class indices
    return decoded_labels[0], confidence_scores[0]

def main():
    # Load the trained LSTM model and LabelEncoder classes
    model_path = "lstm_hand_model.keras"  # Replace with your actual saved model path
    label_classes_path = "label_classes.npy"  # Replace with your saved LabelEncoder classes path
    
    model = load_model(model_path)
    print(f"Model loaded successfully from '{model_path}'")
    
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(label_classes_path, allow_pickle=True)
    print(f"LabelEncoder classes loaded successfully from '{label_classes_path}'")
    
    # Initialize webcam feed
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return

    print("Starting real-time sign language prediction...")
    
    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=2,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame from webcam.")
                    break
                
                # Flip the frame horizontally for a mirrored view
                frame = cv2.flip(frame, 1)
                
                # Convert the frame to RGB as Mediapipe expects RGB input
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process the frame with Mediapipe Hand Detection
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    landmarks_list = []
                    
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Extract hand landmarks (x, y, z coordinates)
                        landmarks = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                        landmarks_list.extend(landmarks)  # Combine all landmarks
                    
                    if len(landmarks_list) == 42:  # Ensure we have both hands' landmarks (21 points per hand)
                        preprocessed_data = preprocess_hand_landmarks(landmarks_list)
                        
                        # Make prediction and get confidence score
                        predicted_label, confidence_score = predict_with_model(model=model,
                                                                               data=preprocessed_data,
                                                                               label_encoder=label_encoder)
                        
                        overlay_text = f"Gesture: {predicted_label} | Confidence: {confidence_score:.2f}"
                        print(overlay_text)
                        
                        # Add overlay text to the frame
                        cv2.putText(frame, overlay_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=1, color=(0, 255, 0), thickness=2)
                    
                    # Draw hand landmarks on the frame
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Display the frame with annotations and overlay text
                cv2.imshow("Sign Language Recognition", frame)

                # Break loop on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nReal-time prediction stopped.")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
