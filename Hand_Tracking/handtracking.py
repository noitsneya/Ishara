import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Define different colors for left and right hands
left_hand_color = (0, 0, 255)  # Red for left hand
right_hand_color = (0, 255, 0)  # Green for right hand

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # To improve performance, optionally mark the image as not writeable
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        # Draw the hand annotations on the image
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Initialize empty lists for left and right hand landmarks
        left_hand_landmarks = None
        right_hand_landmarks = None
        
        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get handedness information (left or right hand)
                handedness = results.multi_handedness[idx].classification[0].label
                
                # Set color based on handedness
                if handedness == "Left":
                    color = left_hand_color
                    left_hand_landmarks = hand_landmarks
                    # Add text label for left hand
                    cv2.putText(image, "Left Hand", 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:  # Right hand
                    color = right_hand_color
                    right_hand_landmarks = hand_landmarks
                    # Add text label for right hand
                    cv2.putText(image, "Right Hand", 
                                (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                 int(hand_landmarks.landmark[0].y * image.shape[0] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Draw landmarks with custom color
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing.DrawingSpec(color=color, thickness=2)
                )
                
                # Extract landmark coordinates (for your model)
                landmarks = []
                for lm in hand_landmarks.landmark:
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    landmarks.append([cx, cy, lm.z])
                
                # Print handedness and first landmark for demonstration
                print(f"{handedness} Hand - First landmark: {landmarks[0]}")
        
        # Display the resulting frame
        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
