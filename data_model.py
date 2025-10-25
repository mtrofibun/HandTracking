import cv2
import mediapipe as mp
import csv
import os
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils


csv_file = "gestures_data.csv"
if not os.path.exists(csv_file):
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        header = [f"{axis}{i}" for i in range(21) for axis in ("x","y","z")]
        header.append("label")
        writer.writerow(header)

sample_counts = {
    "closed" : 0,
    "pointer" : 0,
    "peace" : 0,
    "call" : 0,
    "rocker" : 0,
    "thumbs_up" : 0,
    "open" : 0,
    #additional signals for now
    "sideways" : 0
}

max_samples = 350



cap = cv2.VideoCapture(0)

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    norm = []
    for lm in landmarks:

        norm.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

    scale = ((landmarks[12].x - wrist.x)**2 + (landmarks[12].y - wrist.y)**2)**0.5
    norm = [[x/scale, y/scale, z/scale] for x,y,z in norm]
    return norm
csv_file = open("gestures_data.csv", mode="a", newline="")
csv_writer = csv.writer(csv_file)
while cap.isOpened():
    sucess, image = cap.read()
    if not sucess:
        break

    h, w, c = image.shape
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('1'):
                gesture = "closed"
            elif key == ord('2'):
                gesture = "pointer"
            elif key == ord('3'):
                gesture = "peace"
            elif key == ord('4'):
                gesture = "call"
            elif key == ord('5'):
                #pointer and rocker keep interfere with one another
                gesture = "rocker"
            elif key == ord('6'):
                gesture = "thumbs_up"
            elif key == ord('7'):
                gesture = "open"
            else:
                gesture = None

            if gesture and sample_counts[gesture] < max_samples:

                lm_list = [coord for lm in normalize_landmarks(hand_landmarks.landmark) for coord in lm]
                csv_writer.writerow(lm_list + [gesture])

                sample_counts[gesture] += 1

                print(f"{gesture}: {sample_counts[gesture]}/{max_samples}")

                if sample_counts[gesture] >= max_samples:
                    print(f"{gesture} recording complete!")

    cv2.imshow("Collecting data", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
csv_file.close()
cv2.destroyAllWindows()