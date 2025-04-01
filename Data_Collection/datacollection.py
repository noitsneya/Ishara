import pandas as pd
import cv2
import mediapipe as mp
import csv
import os
import numpy as np

# Initialize directories
os.makedirs("datasets", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# CSV setup (adjusted path)
output_file = "datasets/isl_data.csv"

# Create CSV with header if new
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Create header with separate columns for left and right hands
        header = ["label"]
        for hand in ["left", "right"]:
            for i in range(21):
                for coord in ("x", "y", "z"):
                    header.append(f"{hand}_{coord}{i}")
        writer.writerow(header)

# Camera setup
cap = cv2.VideoCapture(0)
current_label = None

# Define colors for visualization
left_hand_color = (0, 0, 255)  # Red for left hand
right_hand_color = (0, 255, 0)  # Green for right hand

def collect_data_for_label(label):
    recording = False
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # MediaPipe processing
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Initialize empty arrays for left and right hand landmarks
            left_hand_landmarks = np.zeros(21 * 3)  # 21 landmarks, 3 coordinates each
            right_hand_landmarks = np.zeros(21 * 3)
            
            # Status text for display
            status_text = f"Label: {label} | Recording: {'ON' if recording else 'OFF'}"
            hands_text = "No hands detected"
            
            if results.multi_hand_landmarks:
                hands_detected = []
                
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get handedness information (left or right hand)
                    handedness = results.multi_handedness[idx].classification[0].label
                    confidence = results.multi_handedness[idx].classification[0].score
                    hands_detected.append(handedness)
                    
                    # Set color based on handedness
                    if handedness == "Left":
                        color = left_hand_color
                        # Extract landmarks for left hand
                        landmarks = []
                        for i, lm in enumerate(hand_landmarks.landmark):
                            landmarks.extend([lm.x, lm.y, lm.z])
                        left_hand_landmarks = np.array(landmarks)
                        
                        # Add text label for left hand
                        cv2.putText(image, f"Left Hand ({confidence:.2f})", 
                                    (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                     int(hand_landmarks.landmark[0].y * image.shape[0] - 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    else:  # Right hand
                        color = right_hand_color
                        # Extract landmarks for right hand
                        landmarks = []
                        for i, lm in enumerate(hand_landmarks.landmark):
                            landmarks.extend([lm.x, lm.y, lm.z])
                        right_hand_landmarks = np.array(landmarks)
                        
                        # Add text label for right hand
                        cv2.putText(image, f"Right Hand ({confidence:.2f})", 
                                    (int(hand_landmarks.landmark[0].x * image.shape[1]), 
                                     int(hand_landmarks.landmark[0].y * image.shape[0] - 20)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Draw landmarks with custom color
                    mp_drawing.draw_landmarks(
                        image,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing_styles.get_default_hand_landmarks_style(),
                        mp_drawing.DrawingSpec(color=color, thickness=2)
                    )
                
                hands_text = f"Detected: {', '.join(hands_detected)}"
                
                # Record data if recording is enabled
                if recording:
                    # Combine left and right hand data
                    combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
                    
                    with open(output_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([label] + combined_landmarks.tolist())
                        print(f"Collected data for {label} with {hands_text}")

            # Display status information
            cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(image, "Press SPACE to toggle recording, ESC to exit", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Toggle recording state
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                recording = not recording
                print(f"Recording {'started' if recording else 'stopped'} for {label}")

            cv2.imshow("ISL Data Collector", image)
            if key == 27:  # ESC to exit current session
                break

# Main loop
try:
    while True:
        new_label = input("Enter letter label (A-Z, empty to keep previous): ").upper()
            
        if new_label:
            current_label = new_label
        elif not current_label:
            print("Please enter a label first!")
            continue
            
        collect_data_for_label(current_label)
        cv2.destroyAllWindows()

except KeyboardInterrupt:
    print("\nCleaning up dataset...")
    df = pd.read_csv(output_file)
    df = df.drop_duplicates()
    df.to_csv(output_file, index=False)
    print(f"Saved {len(df)} unique samples to {output_file}")
    print("\nExiting program...")
        
finally:
    cap.release()
