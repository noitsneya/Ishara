import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Load the entire processing pipeline
try:
    pipeline = joblib.load('models\hand_gesture_pipeline_01-04-2025-22-35-00.joblib')
    print("Model pipeline loaded successfully")
except FileNotFoundError:
    pipeline = None
    print("Warning: Model pipeline not found. Prediction disabled.")

cap = cv2.VideoCapture(0)

current_prediction = None
confidence_counts = {}
threshold = 3  # Number of consecutive matches needed

def preprocess_landmarks(landmarks):
    """Convert MediaPipe landmarks to pipeline-compatible format"""
    # Extract coordinates in correct order (x0,y0,z0, x1,y1,z1,...)
    return np.array([[lm.x, lm.y, lm.z for lm in landmarks.landmark]]).flatten()

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        # Image processing
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract and preprocess features
                raw_features = preprocess_landmarks(hand_landmarks)
                
                if pipeline is not None:
                    try:
                        # The pipeline handles normalization internally
                        prediction = pipeline.predict([raw_features])[0]
                        confidence = np.max(pipeline.predict_proba([raw_features]))
                        
                        # Stability check
                        if current_prediction != prediction:
                            confidence_counts[prediction] = confidence_counts.get(prediction, 0) + 1
                            
                            if confidence_counts[prediction] >= threshold:
                                current_prediction = prediction
                                confidence_counts = {}
                        else:
                            confidence_counts = {}  # Reset if same prediction

                        # Display prediction with confidence
                        display_text = f'{current_prediction} ({confidence:.2f})'
                        cv2.putText(image, display_text, (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    except Exception as e:
                        cv2.putText(image, f'Error: {str(e)}', (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                )

        cv2.imshow('Sign Language Recognition', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
