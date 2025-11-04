import webview, joblib
from flask import Flask, render_template, request, jsonify,Response
import threading,os,json,cv2,pyautogui, math
import mediapipe as mp
import numpy as np
from pyparsing import results
SAVE_FILE = 'saved_data.json'

# need undo and redo gesture
# add hide camera button
# can do pinky, maybe thumbs up by making thumb index of y greater than all the indexs
# reassign variables when saving not when active in the camera to reduce copies

webcam_off = False



keybinds = {}


# camera
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model = joblib.load("gesture_model.pkl")

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cam = cv2.VideoCapture(0)

frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

def strip_input(keybindValue):
    keybindValue = keybindValue.lower()
    if len(keybindValue) > 1:
        keybindValue = keybindValue.split(" ")
    return keybindValue

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    norm = []
    for lm in landmarks:
        norm.append([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])
    scale = ((landmarks[12].x - wrist.x)**2 + (landmarks[12].y - wrist.y)**2) ** 0.5
    norm = [[x/scale,y/scale,x/scale] for x,y,z in norm]
    return norm

def enable_keybind(value):
        if isinstance(value, list):
            if len(value) >= 3:
                pyautogui.hotkey(value[0], value[1],value[2])
            else:
                pyautogui.hotkey(value[0], value[1])
        else:
            pyautogui.press(value)


def track_distance(p1,p2):
    distance = math.sqrt(
        (p1.x - p2.x)**2 +
        (p1.y - p2.y)**2 +
        (p1.z - p2.z)**2)
    return distance

def is_pinch(hand_landmarks):
    landmarks = hand_landmarks.landmark
    pointer_tip = landmarks[8]
    thumb_tip = landmarks[4]

    FINGERS1 = [12,16,20,8]
    FINGERS2 = [9, 13, 17,5]

    middle_down = landmarks[FINGERS1[0]].y > landmarks[FINGERS2[0]].y
    ring_down = landmarks[FINGERS1[1]].y > landmarks[FINGERS2[1]].y
    pinky_down = landmarks[FINGERS1[2]].y > landmarks[FINGERS2[2]].y
    pointer_up = landmarks[FINGERS1[3]].y > landmarks[FINGERS2[3]].y

    distance = track_distance(pointer_tip,thumb_tip)
    if distance < 0.06 and middle_down and ring_down and pinky_down and not pointer_up:
        return True
    return False

def is_pinch_2(hand_landmarks):
    landmarks = hand_landmarks.landmark
    pointer_tip = landmarks[8]
    thumb_tip = landmarks[4]

    FINGERS1 = [12, 16, 20, 8]
    FINGERS2 = [9, 13, 17, 5]

    middle_down = landmarks[FINGERS1[0]].y > landmarks[FINGERS2[0]].y
    ring_down = landmarks[FINGERS1[1]].y > landmarks[FINGERS2[1]].y
    pinky_down = landmarks[FINGERS1[2]].y > landmarks[FINGERS2[2]].y
    pointer_up = landmarks[FINGERS1[3]].y < landmarks[FINGERS2[3]].y

    distance = track_distance(pointer_tip, thumb_tip)
    if 0.07 < distance < 0.13 and middle_down and ring_down and pinky_down and pointer_up:
        return True
    return False


pinch1Check,pinch2Check = 0,0

def gen_frames():
    global pinch1Check, pinch2Check

    cam = cv2.VideoCapture(0)

    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while cam.isOpened():
            ret, frame = cam.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                #output works we just need to make sure it doesn't overlap pinch1 + 2 + pinky??

                if not is_pinch(hand_landmarks) and not is_pinch_2(hand_landmarks):
                    norm_landmarks = normalize_landmarks(hand_landmarks.landmark)
                    row = [coord for lm in norm_landmarks for coord in lm]
                    x = np.array(row).reshape(1, -1)
                    output = model.predict(x)[0]
                    # keybinds doesn't have scroll we can add that for it to be easier
                    print(keybinds)
                    if output != "open":

                        if keybinds['finalKeybinds'][output]["scroll"] != '0':
                            print("scrolling")
                            scrollValue = int(keybinds["finalKeybinds"][output]["scroll"])
                            print(scrollValue)
                            pyautogui.scroll(scrollValue)
                        else:
                            newValue = strip_input(keybinds["finalKeybinds"][output]["value"])
                            enable_keybind(newValue)

                        cv2.putText(image, f"{output} sign : {keybinds['finalKeybinds'][output]['value']}", (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)



                if is_pinch(hand_landmarks):

                    if keybinds["finalKeybinds"]['pinch1']['scroll'] != '0':
                        scrollValue = int(keybinds["finalKeybinds"]['pinch1']["scroll"])
                        print(scrollValue)
                        pyautogui.scroll(scrollValue)
                    else:
                        newPinchValue = strip_input(keybinds["finalKeybinds"]['pinch1']['value'])
                        pinch1Check += 1
                        enable_keybind(newPinchValue)
                    cv2.putText(image, f"Pinch 1 : {keybinds['finalKeybinds']['pinch1']['value']}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    pinch1Check = 0

                if is_pinch_2(hand_landmarks):
                    if keybinds['finalKeybinds']['pinch2']['scroll'] != '0':
                        scrollValue = int(keybinds["finalKeybinds"]['pinch2']["scroll"])
                        print(scrollValue)
                        pyautogui.scroll(scrollValue)
                    else:
                        newPinchValue = strip_input(keybinds['finalKeybinds']['pinch2']['value'])
                        pinch1Check += 1
                        enable_keybind(newPinchValue)
                    cv2.putText(image, f"Pinch 2 : {keybinds['finalKeybinds']['pinch2']['value']}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    pinch2Check = 0


            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()



cv2.destroyAllWindows()

#routes


if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE) as f:
        keybinds = json.load(f)
        print("Initial keybinds loaded:", keybinds)


def save(binds):
    with open(SAVE_FILE, 'w') as f:
        json.dump(binds, f)

    global keybinds
    keybinds = binds
    print("saved keybinds:", keybinds)


def load():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            data = json.load(f)
            global keybinds
            keybinds = data
            print("loaded keybinds:", keybinds)
            return data

    return {"finalKeybinds": {}}

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/load')
def load_route():
    data = load()
    print("load")
    return jsonify(data["finalKeybinds"])


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/buttonclick', methods=['POST'])
def button_click():
    data = request.get_json()
    save(data)
    message = "Changes are saved"
    print('Changes have been saved')
    return jsonify({"message": message})


@app.route('/webcam', methods=['POST'])
def webcam():
    global webcam_off
    webcam_off = request.data.decode('utf-8')
    print(f"Webcam toggled: {webcam_off}")
    return jsonify({"status": "ok"})


def start_flask():
    app.run(host="127.0.0.1", port=5000)


if __name__ == '__main__':
    threading.Thread(target=start_flask, daemon=True).start()
    webview.create_window('Handtracking Application', "http://127.0.0.1:5000/")
    webview.start()