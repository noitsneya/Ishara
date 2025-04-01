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
current_label = input("Enter letter label (A-Z): ").upper()

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
                
                # Save landmarks on 's' key press
                if cv2.waitKey(1) & 0xFF == ord("s"):
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    with open(output_file, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([current_label] + landmarks)
                    print(f"Saved sample for {current_label}")

        cv2.imshow("ISL Data Collector", image)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
