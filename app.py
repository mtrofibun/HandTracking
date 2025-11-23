import webview, joblib, os, sys, shutil, time, threading, os, json, cv2, pyautogui, math
from flask import Flask, render_template, request, jsonify, Response
import mediapipe as mp
import numpy as np
import psutil
import sklearn
import sklearn.ensemble._forest
import sklearn.preprocessing
import sklearn.svm
import sklearn.utils
import sklearn.base
from pyparsing import results

def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

DEFAULT_FILE = resource_path('saved_data.json')  # original defaults
USER_FILE = os.path.join(os.path.expanduser('~'), 'saved_data.json')

webcam_off = False

if not os.path.exists(USER_FILE):
    shutil.copy(DEFAULT_FILE, USER_FILE)

keybinds = {}

# camera
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

model_path = resource_path("gesture_model.pkl")
model = joblib.load(model_path)

def lower_priority():
    """Lower process priority to reduce CPU impact"""
    try:
        p = psutil.Process()
        if sys.platform == 'win32':
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        else:
            p.nice(10)  # Unix nice value
        print("Process priority lowered")
    except:
        pass

def calculate_scroll(value,number):
    value = value.split(" ")
    if value[1] == 'out':
        number = number * -1
    return number

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
previousState = 'none'
currentState = 'none'
gestureStartTime = 0
THRESHOLD = 0.1

global_cam = None
cam_lock = threading.Lock()


def initialize_camera():
    global global_cam

    with cam_lock:
        if global_cam is not None and global_cam.isOpened():
            print("Camera already initialized")
            return global_cam

        cam = None

        print("Trying DirectShow backend...")
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if cam.isOpened():
            print("Camera opened with DirectShow!")
        else:
            cam.release()

            print("Trying default backend...")
            cam = cv2.VideoCapture(0)
            if cam.isOpened():
                print("Camera opened with default backend!")
            else:
                cam.release()

                print("Trying MSMF backend...")
                cam = cv2.VideoCapture(0, cv2.CAP_MSMF)
                if cam.isOpened():
                    print("Camera opened with MSMF!")
                else:
                    cam.release()
                    cam = None

        if cam is not None and cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cam.set(cv2.CAP_PROP_FPS, 30)
            print(
                f"Camera properties: {cam.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cam.get(cv2.CAP_PROP_FRAME_HEIGHT)} @ {cam.get(cv2.CAP_PROP_FPS)}fps")
            global_cam = cam

        return cam


def gen_frames():
    global pinch1Check, pinch2Check, currentState, gestureStartTime, THRESHOLD, previousState, global_cam

    cam = initialize_camera()

    if cam is None or not cam.isOpened():
        return





    with mp_hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            model_complexity=1,
        static_image_mode = False
    ) as hands:

        while cam.isOpened():
            with cam_lock:
                ret, frame = cam.read()

            if not ret:
                print("Failed to read frame")
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]


                norm_landmarks = normalize_landmarks(hand_landmarks.landmark)
                row = [coord for lm in norm_landmarks for coord in lm]
                x = np.array(row).reshape(1, -1)
                output = model.predict(x)[0]

                if is_pinch(hand_landmarks):
                    gesture = 'pinch1'
                elif is_pinch_2(hand_landmarks):
                    gesture = 'pinch2'
                else:
                    gesture = output

                currentTime = time.time()

                if gesture != "open":
                    if gesture != currentState:
                        currentState = gesture
                        gestureStartTime = currentTime
                    else:
                        if currentTime - gestureStartTime >= THRESHOLD:
                            if keybinds['finalKeybinds'][currentState]['repeat'] == 'true':
                                if keybinds['finalKeybinds'][currentState]["scroll"] != '0':
                                    scrollValue = int(keybinds["finalKeybinds"][currentState]["scroll"])
                                    newscrollValue = calculate_scroll(
                                        keybinds['finalKeybinds'][currentState]["value"],
                                        scrollValue)
                                    pyautogui.scroll(newscrollValue)
                                else:
                                    newValue = strip_input(keybinds["finalKeybinds"][currentState]["value"])
                                    enable_keybind(newValue)

                            elif previousState != currentState and keybinds['finalKeybinds'][currentState][
                                'repeat'] == 'false':
                                previousState = currentState
                                if keybinds['finalKeybinds'][previousState]["scroll"] != '0':
                                    scrollValue = int(keybinds["finalKeybinds"][currentState]["scroll"])
                                    newscrollValue = calculate_scroll(keybinds['finalKeybinds'][currentState]["value"],
                                                                      scrollValue)
                                    pyautogui.scroll(newscrollValue)
                                else:
                                    newValue = strip_input(keybinds["finalKeybinds"][currentState]["value"])
                                    enable_keybind(newValue)

                            cv2.putText(image,
                                        f"{currentState} sign : {keybinds['finalKeybinds'][currentState]['value']}",
                                        (50, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            encodeParam = [int(cv2.IMWRITE_JPEG_QUALITY), 60]
            ret, buffer = cv2.imencode('.jpg', image, encodeParam)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    print("gen_frames() ended")


#routes


with open(USER_FILE, 'r') as f:
    keybinds = json.load(f)


def save(binds):
    global keybinds
    keybinds = binds

    with open(USER_FILE, 'w') as f:
        json.dump(binds, f, indent=4)

    print("Saved keybinds:", keybinds)


def load():
    global keybinds

    if os.path.exists(USER_FILE):
        with open(USER_FILE, 'r') as f:
            data = json.load(f)
            keybinds = data
            print("Loaded keybinds:", keybinds)
            return data
    else:
        keybinds = {}
        print("No keybinds file found, starting with empty dict.")
        return keybinds

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
    try:
        return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Error in video_feed: {e}")
        import traceback
        traceback.print_exc()
        return "Error", 500


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
    app.run(host="127.0.0.1", port=5000, debug = False)


if __name__ == '__main__':
    lower_priority()
    try:
        threading.Thread(target=start_flask, daemon=True).start()
        time.sleep(1)
        webview.create_window('Handtracking Application', "http://127.0.0.1:5000/")
        webview.start(gui='edgehtml')
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback

        traceback.print_exc()
        input("Press Enter to exit...")