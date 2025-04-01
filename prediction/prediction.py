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

recent_predictions = []
max_predictions = 10

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
                
                # Make prediction if model is available
                if model is not None:
                    try:
                        prediction = model.predict([features])
                        
                        # Add current prediction to list
                        recent_predictions.append(prediction[0])
                        if len(recent_predictions) > max_predictions:
                            recent_predictions.pop(0)
                        
                        # Get most frequent prediction
                        from collections import Counter
                        most_common = Counter(recent_predictions).most_common(1)[0][0]
                        
                        # Display stable prediction
                        cv2.putText(image, f'Prediction: {most_common}', (10, 30),
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
