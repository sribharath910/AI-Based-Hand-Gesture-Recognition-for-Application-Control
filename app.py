
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("streamlit_app\gesture_model.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

st.title("✋ Hand Gesture Recognition (Web Version)")
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

st.sidebar.title("🛑 Stop Detection")
if st.sidebar.button("Stop"):
    cap.release()
    st.stop()

label_map = {
    0: "Swipe Right",
    1: "Swipe Left",
    2: "Play/Stop",
    3: "Volume Up",
    4: "Volume Down",
    5: "Swipe Up",
    6: "Swipe Down"
}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    gesture_text = "No Hand Detected"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                gesture_text = label_map.get(prediction, "Unknown")
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {gesture_text}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    FRAME_WINDOW.image(frame, channels="BGR")
