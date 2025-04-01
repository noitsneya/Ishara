import pandas as pd
import cv2
import mediapipe as mp
import csv
import os

# Initialize directories
os.makedirs("datasets", exist_ok=True)

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# CSV setup (adjusted path)
output_file = "datasets/isl_data.csv"

# Create CSV with header if new
if not os.path.exists(output_file):
    with open(output_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label"] + [f"{coord}{i}" for i in range(21) for coord in ("x", "y", "z")])

# Camera setup
cap = cv2.VideoCapture(0)
current_label = None

def collect_data_for_label(label):
    recording = False
    with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                continue

            # MediaPipe processing
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    
                    if recording:
                        landmarks = []
                        for lm in hand_landmarks.landmark:
                            landmarks.extend([lm.x, lm.y, lm.z])
                        
                        with open(output_file, mode="a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([label] + landmarks)  # Use passed label
                            print(f"collected data for {label}")

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

    df = pd.read_csv(output_file)
    df = df.drop_duplicates()
    df.to_csv(output_file, index=False)

    print("\nExiting program...")
        
finally:
    cap.release()
