import cv2
import mediapipe as mp

import pyautogui
from pyparsing import results

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)



cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))



while cam.isOpened():
    ret,frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame,1)
    image=  cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

    results = hands.process(image)

    image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
        )
        for id, lm in enumerate(hand_landmarks.landmark):
            x = int(lm.x * frame.shape[1])
            y = int(lm.y * frame.shape[0])
            print(f"Landmark {id}: {lm}")
    cv2.imshow("Handtracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cam.release()
cv2.destroyAllWindows()
