import webview
from flask import Flask, render_template, request, jsonify,Response
import threading,os,json,cv2,pyautogui, math
import mediapipe as mp
from pyparsing import results
SAVE_FILE = 'saved_data.json'

#https://github.com/r0x0r/pywebview/blob/master/examples/localhost_ssl.py

# fix pinch1 and pinch2 not loading correctly for odd reason
# need undo and redo gesture
# add hide camera button
with open(SAVE_FILE) as l:
    file = json.load(l)
    closedvalue = file["finalKeybinds"]["closedOff"]
    pointervalue = file["finalKeybinds"]["pointer"]
    peacevalue = file["finalKeybinds"]["peace"]
    pinch1value = file["finalKeybinds"]["pinch1"]
    pinch2value = file["finalKeybinds"]["pinch2"]
    rockervalue = file["finalKeybinds"]["rocker"]


keybinds = { 'closedOff' : closedvalue,
            'pointer' : pointervalue,
            'peace' : peacevalue,
             'pinch1' : pinch1value,
             'pinch2' : pinch2value,
             'rocker' : rockervalue
            }
print(keybinds)


# camera 
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

def strip_input(keybindValue):
    keybindValue = keybindValue.lower()
    if len(keybindValue) > 1:
        keybindValue = keybindValue.split(" ")
        if keybindValue[0] == 'scroll':
            if keybindValue[1] == 'out':
                keybindValue = '100'
            else:
                keybindValue = '-100'
            keybindValue = int(keybindValue)
    return keybindValue

def enable_keybind(value):
        if isinstance(value, list):
            pyautogui.hotkey(value[0], value[1])
        else:
            pyautogui.press(value)

def is_pointer(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20]
    FINGER_PIPS = [5,9,13,17]

    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_down = landmarks[FINGER_TIPS[1]].y > landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y



    distance = track_distance(landmarks[8], landmarks[4])


    if index_up and middle_down and ring_down and pinky_down and distance > 0.15:
        return True
    return False


def is_peace_sign(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8, 12, 16, 20,4]
    FINGER_PIPS = [6, 10, 14, 18,12]

    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_up = landmarks[FINGER_TIPS[1]].y < landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y
    crossed_thumb = landmarks[FINGER_TIPS[4]].y > landmarks[FINGER_PIPS[4]].y


    if index_up and middle_up and ring_down and pinky_down:
        return True
    return False

def is_closed(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20,8]
    FINGER_PIPS = [5,9,13,17,4]

    index_down = landmarks[FINGER_TIPS[0]].y > landmarks[FINGER_PIPS[0]].y
    middle_down = landmarks[FINGER_TIPS[1]].y > landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y

    if index_down and middle_down and ring_down and pinky_down:
        return True
    return False

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
    if 0.08 < distance < 0.15 and middle_down and ring_down and pinky_down and pointer_up:
        return True
    return False

def is_rocker(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20]
    FINGER_PIPS = [5,9,13,17]

    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_up = landmarks[FINGER_TIPS[1]].y > landmarks[FINGER_PIPS[1]].y
    pointer_up = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y < landmarks[FINGER_PIPS[3]].y
    if index_up and middle_up and pointer_up and pinky_down:
        return True
    return False

peaceCheck,pointerCheck,closedCheck,pinch1Check,pinch2Check,rockerCheck = 0,0,0,0,0,0

def gen_frames():
    global peaceCheck, pointerCheck, closedCheck, pinch1Check, pinch2Check, rockerCheck

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
                if is_peace_sign(hand_landmarks):
                    newPeaceValue = strip_input(keybinds['peace'])
                    if isinstance(newPeaceValue, int):
                        pyautogui.scroll(newPeaceValue)
                    elif peaceCheck == 0:
                        peaceCheck += 1
                        enable_keybind(newPeaceValue)
                    cv2.putText(image, f"Peace sign : {keybinds['peace']}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    peaceCheck = 0

                if is_pointer(hand_landmarks):
                    newPointerValue = strip_input(keybinds['pointer'])
                    if isinstance(newPointerValue, int):
                        pyautogui.scroll(newPointerValue)
                    elif pointerCheck == 0:
                        pointerCheck += 1
                        enable_keybind(newPointerValue)
                    print("Pointer sign detected:", newPointerValue)
                    cv2.putText(image, f"Pointer sign : {keybinds['pointer']}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    pointerCheck = 0

                if is_closed(hand_landmarks):
                    newClosedValue = strip_input(keybinds['closedOff'])
                    if isinstance(newClosedValue, int):
                        pyautogui.scroll(newClosedValue)
                    elif closedCheck == 0:
                        closedCheck += 1
                        enable_keybind(newClosedValue)
                    cv2.putText(image, f"Closed hand : {keybinds['closedOff']}", (50, 150),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    closedCheck = 0

                if is_pinch(hand_landmarks):
                    newPinchValue = strip_input(keybinds['pinch1'])
                    if isinstance(newPinchValue, int):
                        pyautogui.scroll(newPinchValue)
                    elif pinch1Check == 0:
                        pinch1Check += 1
                        enable_keybind(newPinchValue)
                    cv2.putText(image, f"Pinch 1 : {keybinds['pinch1']}", (50, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    pinch1Check = 0

                if is_pinch_2(hand_landmarks):
                    newPinchValue = strip_input(keybinds['pinch2'])
                    if isinstance(newPinchValue, int):
                        pyautogui.scroll(newPinchValue)
                    elif pinch2Check == 0:
                        pinch2Check += 1
                        enable_keybind(newPinchValue)
                    cv2.putText(image, f"Pinch 2 : {keybinds['pinch2']}", (50, 250),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    pinch2Check = 0

                if is_rocker(hand_landmarks):
                    newrockerValue = strip_input(keybinds['rocker'])
                    if isinstance(newrockerValue, int):
                        pyautogui.scroll(newrockerValue)
                    elif rockerCheck == 0:
                        rockerCheck += 1
                        enable_keybind(newrockerValue)
                    cv2.putText(image, f"Rocker : {keybinds['rocker']}", (50, 300),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                else:
                    rockerCheck = 0

            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cam.release()



cv2.destroyAllWindows()

#routes


def save(binds):
    with open(SAVE_FILE,'w') as f:
        json.dump(binds,f)
        keybinds['closedOff'] = binds["finalKeybinds"]["closedOff"]
        keybinds['pointer'] = binds["finalKeybinds"]["pointer"]
        keybinds['peace'] = binds["finalKeybinds"]["peace"]
        keybinds['pinch1'] = binds["finalKeybinds"]["pinch1"]
        keybinds['pinch2'] = binds["finalKeybinds"]["pinch2"]
        keybinds['rocker'] = binds["finalKeybinds"]["rocker"]
        print("saved", binds)
        print("saved", keybinds)

def load():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            data = json.load(f)
            keybinds['closedOff'] = data["finalKeybinds"]["closedOff"]
            keybinds['pointer'] = data["finalKeybinds"]["pointer"]
            keybinds['peace'] = data["finalKeybinds"]["peace"]
            keybinds['pinch1'] = data["finalKeybinds"]["pinch1"]
            keybinds['pinch2']  = data["finalKeybinds"]["pinch2"]
            keybinds['rocker'] = data["finalKeybinds"]["rocker"]
            print("load", keybinds)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/load')
def load_route():
    with open(SAVE_FILE,'r') as f:
        data = json.load(f)
        print("loading", data)
        return data['finalKeybinds']
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/buttonclick',methods=['POST'])
def button_click():
    data = request.get_json()

    save(data)
    message = "Changes are saved"
    print('Changes have been saved')
    return jsonify({"message": message})

@app.route('/webcam')
def webcam():
    pass


def start_flask():
    app.run(host="127.0.0.1", port=5000)

if __name__ == '__main__':
    threading.Thread(target=start_flask, daemon=True).start()

    webview.create_window('Handtracking Application',"http://127.0.0.1:5000/")
    webview.start()