import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Check if model file exists
model = None
try:
    model = joblib.load('models\\random_forest_model.joblib')
    print("Model loaded successfully")
except FileNotFoundError:
    print("Warning: model not found. Prediction functionality will be disabled.")

cap = cv2.VideoCapture(0)

current_prediction = None
confidence_counts = {}
threshold = 3

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Convert the image to RGB and then BGR again - for media pipe
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract features
                features = []
                for id, landmark in enumerate(hand_landmarks.landmark):
                    features.extend([landmark.x, landmark.y, landmark.z])
                
                if model is not None:
                    try:
                        prediction = model.predict([features])
                        pred_value = prediction[0]
                        
                        # Initialize current prediction if needed
                        if current_prediction is None:
                            current_prediction = pred_value
                        
                        # Update confidence counts
                        if pred_value == current_prediction:
                            confidence_counts[pred_value] = 0  # Reset count for current prediction
                        else:
                            # Increment count for this prediction
                            confidence_counts[pred_value] = confidence_counts.get(pred_value, 0) + 1
                            
                            # Change current prediction if threshold reached
                            if confidence_counts[pred_value] >= threshold:
                                current_prediction = pred_value
                                confidence_counts = {}  # Reset all counts
                        
                        # Display stable prediction
                        cv2.putText(image, f'Prediction: {current_prediction}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    except Exception as e:
                        cv2.putText(image, f'Prediction error: {str(e)}', (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

                
                # Draw hand landmarks
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Save features to CSV if needed
                # Uncomment to save data for training
                # with open('hand_features.csv', 'a') as f:
                #     f.write('0,' + ','.join(map(str, features)) + '\n')

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
