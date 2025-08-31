import webview
from flask import Flask, render_template, request, jsonify
import threading,os,json,cv2,pyautogui, time
import mediapipe as mp
from pyparsing import results
SAVE_FILE = 'saved_data.json'

openvalue = 'scroll Y'
closedvalue = 'scroll X'
leftvalue = 'ctrl Z'
rightvalue = 'ctrl X'
pointervalue = 'b'
peacevalue = 'e'

valuearray = [openvalue,closedvalue,leftvalue,rightvalue,pointervalue,peacevalue]

keybinds = { 'opened' : openvalue,
            'closedOff' : closedvalue,
            'left' : leftvalue,
            'right' : rightvalue,
            'pointer' : pointervalue,
            'peace' : peacevalue,
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

def is_pointer(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20]
    FINGER_PIPS = [5,9,13,17]

    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_down = landmarks[FINGER_TIPS[1]].y > landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y

    return index_up and middle_down and ring_down and pinky_down

def strip_input(keybindValue):
    keybindValue = keybindValue.lower()
    if len(keybindValue) > 1:
        keybindValue = keybindValue.split(" ")


    return keybindValue

def is_peace_sign(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8, 12, 16, 20,4]
    FINGER_PIPS = [6, 10, 14, 18,12]

    # Peace sign: index & middle up, ring & pinky down
    index_up = landmarks[FINGER_TIPS[0]].y < landmarks[FINGER_PIPS[0]].y
    middle_up = landmarks[FINGER_TIPS[1]].y < landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y
    crossed_thumb = landmarks[FINGER_TIPS[4]].y > landmarks[FINGER_PIPS[4]].y

    return index_up and middle_up and ring_down and pinky_down and crossed_thumb

def is_closed(hand_landmarks):
    landmarks = hand_landmarks.landmark

    FINGER_TIPS = [8,12,16,20,8]
    FINGER_PIPS = [5,9,13,17,4]

    index_down = landmarks[FINGER_TIPS[0]].y > landmarks[FINGER_PIPS[0]].y
    middle_down = landmarks[FINGER_TIPS[1]].y > landmarks[FINGER_PIPS[1]].y
    ring_down = landmarks[FINGER_TIPS[2]].y > landmarks[FINGER_PIPS[2]].y
    pinky_down = landmarks[FINGER_TIPS[3]].y > landmarks[FINGER_PIPS[3]].y
    #thumb_check = landmarks[FINGER_TIPS[4]].y > landmarks[FINGER_PIPS[4]].y

    return index_down and middle_down and ring_down and pinky_down

peaceCheck,pointerCheck,closedCheck = 0,0,0

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

        if is_peace_sign(hand_landmarks):
            newPeaceValue = strip_input(peacevalue)
            if peaceCheck == 0:
                peaceCheck += 1
                if isinstance(newPeaceValue,list):
                    pyautogui.hotkey(newPeaceValue[0],newPeaceValue[1])
                else:
                    pyautogui.press(newPeaceValue)
                print('Peace sign detected')
                print(newPeaceValue)

            cv2.putText(image, "Peace sign", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        else:
            peaceCheck = 0

        if is_pointer(hand_landmarks):
            cv2.putText(image, "Pointer sign", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newPointerValue = strip_input(pointervalue)
            if pointerCheck == 0:
                pointerCheck += 1
                if isinstance(newPointerValue,list):
                    pyautogui.hotkey(newPointerValue[0],newPointerValue[1])
                else:
                    pyautogui.press(newPointerValue)
                print('Pointer sign detected')
                print(newPointerValue)
        else:
            pointerCheck = 0

        if is_closed(hand_landmarks):
            cv2.putText(image, "Closed hand", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            newClosedValue = strip_input(closedvalue)
            if closedCheck == 0:
                closedCheck += 1
                if isinstance(newClosedValue,list):
                    pyautogui.hotkey(newClosedValue[0],newClosedValue[1])
                else:
                    pyautogui.press(newClosedValue)
                print('Closed hand detected')
                print(newClosedValue)
        else:
            closedCheck = 0
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
            keybinds['opened'] = data.get('opened',openvalue)
            keybinds['closedOff'] = data.get('closedOff',closedvalue)
            keybinds['left'] = data.get('left',leftvalue)
            keybinds['right'] = data.get('right',rightvalue)
            keybinds['pointer'] = data.get('pointer',pointervalue)
            keybinds['peace'] = data.get('peace',peacevalue)

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
