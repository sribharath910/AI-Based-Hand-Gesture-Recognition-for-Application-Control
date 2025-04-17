import joblib
import pyautogui
import cv2
import mediapipe as mp  # Import MediaPipe
import collections
# Load trained model
import time
model = joblib.load("gesture_model.pkl")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing (optional, for visualization)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def extract_landmarks(hand_landmarks):
    """Extracts X, Y, and Z coordinates (63 features)"""
    landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])  # Collect (x, y, z) for all 21 landmarks
    return landmarks

gesture_history = collections.deque(maxlen=  5) 
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Flip for mirror effect
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = extract_landmarks(hand_landmarks)
            prediction = model.predict([landmarks])[0]  # Predict gesture

            gesture_history.append(prediction)  # Store latest prediction

            # Only trigger action if 3 out of the last 5 predictions are the same
            if gesture_history.count(0) > 3:
                print("Gesture: Swipe Right")
                pyautogui.press("right")  # Move PowerPoint slide forward
                gesture_history.clear()
                time.sleep(1)  # Add a delay of 1 second

            elif gesture_history.count(1) > 3:
                print("Gesture: Swipe Left")
                pyautogui.press("left")  # Move PowerPoint slide backward
                gesture_history.clear()
                time.sleep(1)  #Add a delay of 1 second

            elif gesture_history.count(2) > 3:
                print("Gesture: Play/Pause")
                pyautogui.press("space")
                gesture_history.clear()
                time.sleep(1)

            elif gesture_history.count(3) > 3:  # Volume Up
                print("Gesture: Volume Up")
                pyautogui.press("volumeup")
                gesture_history.clear()
            
            elif gesture_history.count(4) > 3:  # Volume Down
                print("Gesture: Volume Down")
                pyautogui.press("volumedown")
                gesture_history.clear()

            elif gesture_history.count(5) > 3:  # Swipe Up
                print("Gesture: Swipe Up")
                pyautogui.press("up")
                gesture_history.clear()
                time.sleep(1)

            elif gesture_history.count(6) > 3:  # Swipe Down
                print("Gesture: Swipe Down")
                pyautogui.press("down")
                gesture_history.clear()
                time.sleep(1)
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()