import joblib
import pyautogui
import cv2
import mediapipe as mp
import collections
import streamlit as st
import numpy as np
import time

# Load the trained model
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing (optional, for visualization)
mp_drawing = mp.solutions.drawing_utils

# Streamlit setup
st.title("Hand Gesture Control App")
st.write("Detect gestures and control the system in real-time.")
FRAME_WINDOW = st.image([])  # Streamlit image placeholder
gesture_history = collections.deque(maxlen=5)  # Store recent gestures

# Webcam capture
cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Extract X, Y, and Z coordinates (63 features)"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # Collect (x, y, z) for all 21 landmarks
    return landmarks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip frame for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = "No Hand Detected"

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            prediction = model.predict([landmarks])[0]  # Predict gesture

            gesture_history.append(prediction)  # Store the latest prediction

            # Only trigger action if 3 out of the last 5 predictions are the same
            if gesture_history.count(0) > 3:
                gesture_text = "Swipe Right"
                pyautogui.press("right")  # Move PowerPoint slide forward
                gesture_history.clear()
                time.sleep(1)  # Add a delay of 1 second

            elif gesture_history.count(1) > 3:
                gesture_text = "Swipe Left"
                pyautogui.press("left")  # Move PowerPoint slide backward
                gesture_history.clear()
                time.sleep(1)

            elif gesture_history.count(2) > 3:
                gesture_text = "Play/Pause"
                pyautogui.press("space")  # Play/Pause media
                gesture_history.clear()
                time.sleep(1)

            elif gesture_history.count(3) > 3:
                gesture_text = "Volume Up"
                pyautogui.press("volumeup")  # Volume up
                gesture_history.clear()

            elif gesture_history.count(4) > 3:
                gesture_text = "Volume Down"
                pyautogui.press("volumedown")  # Volume down
                gesture_history.clear()

            elif gesture_history.count(5) > 3:
                gesture_text = "Swipe Up"
                pyautogui.press("up")  # Scroll up
                gesture_history.clear()
                time.sleep(1)

            elif gesture_history.count(6) > 3:
                gesture_text = "Swipe Down"
                pyautogui.press("down")  # Scroll down
                gesture_history.clear()
                time.sleep(1)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the predicted gesture on the image
    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
cv2.destroyAllWindows()
