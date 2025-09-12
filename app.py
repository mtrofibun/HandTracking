import webview
from flask import Flask, render_template, request, jsonify
import threading,os,json,cv2,pyautogui, math
import mediapipe as mp
from pyparsing import results
SAVE_FILE = 'saved_data.json'

closedvalue = 'ctrl z'
leftvalue = 'ctrl z'
rightvalue = 'ctrl x'
pointervalue = 'b'
peacevalue = 'e'
pinch1value = 'scroll in'
pinch2value = 'scroll out'
threevalue = 'i'

valuearray = [closedvalue,leftvalue,rightvalue,pointervalue,peacevalue,pinch1value,pinch2value]

keybinds = { 'closedOff' : closedvalue,
            'pointer' : pointervalue,
            'peace' : peacevalue,
             'pinch1' : pinch1value,
             'pinch2' : pinch2value,
             'three' : threevalue
            }
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
                keybindValue = '120'
            else:
                keybindValue = '-120'
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

    if index_up and middle_down and ring_down and pinky_down:
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

    if index_up and middle_up and ring_down and pinky_down and crossed_thumb:
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

    FINGERS1 = [12,16,20]
    FINGERS2 = [9, 13, 17]

    middle_up = landmarks[FINGERS1[0]].y < landmarks[FINGERS2[0]].y
    ring_up = landmarks[FINGERS1[1]].y < landmarks[FINGERS2[1]].y
    pinky_up = landmarks[FINGERS1[2]].y < landmarks[FINGERS2[2]].y

    distance = track_distance(pointer_tip,thumb_tip)
    if distance < 0.06:
        return middle_up and ring_up and pinky_up
    return False

def is_pinch_2(hand_landmarks):
    landmarks = hand_landmarks.landmark
    pointer_tip = landmarks[8]
    thumb_tip = landmarks[4]

    FINGERS1 = [12, 16, 20]
    FINGERS2 = [9, 13, 17]

    middle_up = landmarks[FINGERS1[0]].y < landmarks[FINGERS2[0]].y
    ring_up = landmarks[FINGERS1[1]].y < landmarks[FINGERS2[1]].y
    pinky_up = landmarks[FINGERS1[2]].y < landmarks[FINGERS2[2]].y

    distance = track_distance(pointer_tip, thumb_tip)
    if 0.08 < distance < 0.20 and middle_up and ring_up and pinky_up:
        return True
    return False

def is_three(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20]
    FINGER_PIPS = [5,9,13,17]

    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_up = landmarks[FINGER_TIPS[1]].y < landmarks[FINGER_PIPS[1]].y
    pointer_up = landmarks[FINGER_TIPS[2]].y < landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y
    if index_up and middle_up and pointer_up and pinky_down:
        return True
    return False

peaceCheck,pointerCheck,closedCheck,pinch1Check,pinch2Check,threeCheck = 0,0,0,0,0,0

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
        pinchCheckPoint = track_distance(hand_landmarks.landmark[8], hand_landmarks.landmark[4])
        if is_peace_sign(hand_landmarks):
            newPeaceValue = strip_input(peacevalue)
            if isinstance(newPeaceValue, int):
                pyautogui.scroll(newPeaceValue)
            else:
                if peaceCheck == 0:
                    peaceCheck += 1
                    enable_keybind(newPeaceValue)
                    print('Peace Sign detected')
                    print(newPeaceValue)

            cv2.putText(image, "Peace sign", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            peaceCheck = 0

        if is_pointer(hand_landmarks):
            cv2.putText(image, "Pointer sign", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newPointerValue = strip_input(pointervalue)
            if isinstance(newPointerValue, int):
                pyautogui.scroll(newPointerValue)
            else:
                if pointerCheck == 0:
                    pointerCheck += 1
                    enable_keybind(newPointerValue)
                    print('Pointer sign detected')
                    print(newPointerValue)
                    print(pointerCheck)
        else:
            pointerCheck = 0

        if is_closed(hand_landmarks) :
            cv2.putText(image, "Closed hand", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newClosedValue = strip_input(closedvalue)
            if isinstance(newClosedValue, int):
                pyautogui.scroll(newClosedValue)
            else:
                if closedCheck == 0:
                    closedCheck += 1
                    enable_keybind(newClosedValue)
                print('Closed hand detected')
                print(newClosedValue)
        else:
            closedCheck = 0

        if is_pinch(hand_landmarks):
            cv2.putText(image, "Pinch sign", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newPinchValue = strip_input(pinch1value)
            if isinstance(newPinchValue, int):
                pyautogui.scroll(newPinchValue)
            else:
                if pinch1Check == 0:
                    pinch1Check += 1
                    enable_keybind(newPinchValue)
                print('Pinch1 hand detected')
                print(newPinchValue)
        else:
            pinch1Check = 0


        if is_pinch_2(hand_landmarks):
            cv2.putText(image, "Pinch sign 2", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newPinchValue = strip_input(pinch2value)
            if isinstance(newPinchValue, int):
                pyautogui.scroll(newPinchValue)
            else:
                if pinch2Check == 0:
                    pinch2Check += 1
                    enable_keybind(newPinchValue)
                print('Pinch2 hand detected')
                print(newPinchValue)
        else:
            pinch2Check = 0

        if is_three(hand_landmarks):
            cv2.putText(image, "Three", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newThreeValue = strip_input(threevalue)
            if isinstance(newThreeValue, int):
                pyautogui.scroll(newThreeValue)
            else:
                if threeCheck == 0:
                    threeCheck += 1
                    enable_keybind(newThreeValue)
        else:
            threeCheck = 0
        # checking for hand movements
        #for id, lm in enumerate(hand_landmarks.landmark):
        #    x = int(lm.x * frame.shape[1])
        #    y = int(lm.y * frame.shape[0])
        #    print(f"Landmark {id}: {lm}")
    cv2.imshow("Handtracking", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break




cam.release()
cv2.destroyAllWindows()

#routes


def save(binds):
    with open(SAVE_FILE,'w') as f:
        json.dump(binds,f)


def load():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            data = json.load(f)
            keybinds['closedOff'] = data.get('closedOff',closedvalue)
            keybinds['left'] = data.get('left',leftvalue)
            keybinds['right'] = data.get('right',rightvalue)
            keybinds['pointer'] = data.get('pointer',pointervalue)
            keybinds['peace'] = data.get('peace',peacevalue)
            keybinds['pinch1'] = data.get('pinch1',pinch1value)
            keybinds['pinch2'] = data.get('pinch2', pinch2value)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/load')
def load_route():
    return jsonify(keybinds)

@app.route('/api/buttonclick',methods=['POST'])
def button_click():
    data = request.get_json()
    for keys in keybinds.keys():
        keys.value = data.value

    save(keybinds)
    message = "Changes are saved"
    print('Changes have been saved')
    return jsonify({"message": message})
