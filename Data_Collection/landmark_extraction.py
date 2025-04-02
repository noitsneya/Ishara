import pandas as pd
import cv2
import mediapipe as mp
import csv
import os
import numpy as np
import time

# Initialize directories
os.makedirs("datasets", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# CSV setup
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

# Define colors for visualization
left_hand_color = (0, 0, 255)  # Red for left hand
right_hand_color = (0, 255, 0)  # Green for right hand
recording_color = (0, 0, 255)   # Red for recording
cooldown_color = (0, 255, 255)  # Yellow for cooldown

# List of 10 words for labels
word_labels = ["HELLO", "THANK YOU", "YES", "NO", "PLEASE", "SORRY", "GOOD", "BAD", "LOVE", "FRIEND"]

def collect_data_with_timer():
    current_word_idx = 0
    
    # Timer settings
    recording_duration = 20  # seconds
    cooldown_duration = 10   # seconds
    
    with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5, 
        min_tracking_confidence=0.5,
        max_num_hands=2) as hands:
        
        while current_word_idx < len(word_labels):
            current_label = word_labels[current_word_idx]
            
            # Start recording timer
            recording_start_time = time.time()
            recording_end_time = recording_start_time + recording_duration
            is_recording = True
            is_cooldown = False
            
            print(f"Started recording for label: {current_label}")
            
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    continue

                # Get current time for timer calculations
                current_time = time.time()
                
                # Check if recording phase has ended
                if is_recording and current_time >= recording_end_time:
                    is_recording = False
                    is_cooldown = True
                    cooldown_start_time = current_time
                    cooldown_end_time = cooldown_start_time + cooldown_duration
                    print(f"Recording ended for {current_label}. Cooldown started.")
                
                # Check if cooldown phase has ended
                if is_cooldown and current_time >= cooldown_end_time:
                    is_cooldown = False
                    current_word_idx += 1
                    print(f"Cooldown ended. Moving to next word.")
                    break
                
                # MediaPipe processing
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Initialize empty arrays for left and right hand landmarks
                left_hand_landmarks = np.zeros(21 * 3)  # 21 landmarks, 3 coordinates each
                right_hand_landmarks = np.zeros(21 * 3)
                
                # Calculate remaining time
                if is_recording:
                    time_remaining = max(0, int(recording_end_time - current_time))
                    status_text = f"Label: {current_label} | RECORDING: {time_remaining}s"
                    phase_color = recording_color
                else:
                    time_remaining = max(0, int(cooldown_end_time - current_time))
                    status_text = f"Label: {current_label} | COOLDOWN: {time_remaining}s"
                    phase_color = cooldown_color
                
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
                    
                    # Record data if in recording phase
                    if is_recording:
                        # Combine left and right hand data
                        combined_landmarks = np.concatenate([left_hand_landmarks, right_hand_landmarks])
                        
                        with open(output_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([current_label] + combined_landmarks.tolist())

                # Display status information
                cv2.putText(image, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, hands_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Word counter
                word_counter = f"Word {current_word_idx + 1}/{len(word_labels)}"
                cv2.putText(image, word_counter, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Progress bar
                bar_width = 300
                
                if is_recording:
                    progress = 1 - (recording_end_time - current_time) / recording_duration
                    bar_color = recording_color
                else:
                    progress = 1 - (cooldown_end_time - current_time) / cooldown_duration
                    bar_color = cooldown_color
                
                filled_width = int(bar_width * progress)
                cv2.rectangle(image, (10, 120), (10 + bar_width, 140), (100, 100, 100), -1)
                cv2.rectangle(image, (10, 120), (10 + filled_width, 140), bar_color, -1)
                
                # Display instructions during cooldown
                if is_cooldown:
                    cv2.putText(image, "Press SPACE to skip to next word, or wait for cooldown", (10, 170), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # Process key presses
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit
                    print("Program terminated by user.")
                    return
                elif key == ord(" ") and is_cooldown:
                    # Skip cooldown and move to next word
                    print(f"Skipping cooldown. Moving to next word.")
                    current_word_idx += 1
                    break

                cv2.imshow("ISL Data Collector", image)

        print("All words completed!")

# Main execution
try:
    print(f"Starting data collection for {len(word_labels)} words: {', '.join(word_labels)}")
    collect_data_with_timer()
    
except KeyboardInterrupt:
    print("\nCleaning up dataset...")
    
finally:
    # Clean up dataset to remove duplicates
    if os.path.exists(output_file):
        df = pd.read_csv(output_file)
        df = df.drop_duplicates()
        df.to_csv(output_file, index=False)
        print(f"Saved {len(df)} unique samples to {output_file}")
    
    print("\nExiting program...")
    cap.release()
    cv2.destroyAllWindows()
