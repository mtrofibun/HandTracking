import cv2
import mediapipe as mp
import joblib
import numpy as np

# Load model
model = joblib.load("gesture_model.pkl")

# Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    norm = []
    for lm in landmarks:
        norm.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = ((landmarks[12].x - wrist.x)**2 + (landmarks[12].y - wrist.y)**2)**0.5
    norm = [[x/scale, y/scale, z/scale] for x,y,z in norm]
    return norm

gesture_to_key = {
    "Peace": "a",
    "ThumbsUp": "space",
    "Fist": "ctrl",
}

while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            norm_landmarks = normalize_landmarks(hand_landmarks.landmark)
            row = [coord for lm in norm_landmarks for coord in lm]

            X = np.array(row).reshape(1, -1)
            gesture = model.predict(X)[0]

            cv2.putText(image, gesture, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Gesture Control", image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()