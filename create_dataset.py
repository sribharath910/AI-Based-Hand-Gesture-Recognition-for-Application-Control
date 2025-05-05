import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)


# Ask user for the gesture name
gesture_name = input("Enter gesture name (e.g., swipe_right, swipe_left, stop, play): ").strip()
data = []

print(f"Collecting data for gesture: {gesture_name}. Please perform the gesture.")
time.sleep(3)  # 3-second countdown before data collection starts
# Open webcam
cap = cv2.VideoCapture(0)

while len(data) < 300:  # Collect 300 samples per gesture
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract 21 landmark points (x, y, z)
            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
            data.append(landmarks.tolist())

            # Draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Collection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
        break

cap.release()
cv2.destroyAllWindows()

# Save collected data to CSV
df = pd.DataFrame(data)
df.to_csv(f"{gesture_name}.csv", index=False)
print(f"Dataset saved as {gesture_name}.csv")
