import webview
from flask import Flask, render_template, request, jsonify
import threading,os,json,cv2,pyautogui
import mediapipe as mp
from pyparsing import results
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

#routes
SAVE_FILE = 'saved_data.json'

openvalue = 'Scroll Y'
closedvalue = 'Scroll X'
leftvalue = 'Ctrl + Z'
rightvalue = 'Ctrl + X'
fingervalue = 'B'
peacevalue = 'E'

valuearray = [openvalue,closedvalue,leftvalue,rightvalue,fingervalue,peacevalue]

keybinds = { 'open' : openvalue,
            'closed' : closedvalue,
            'left' : leftvalue,
            'right' : rightvalue,
            'finger' : fingervalue,
            'peace' : peacevalue,
            }

def save(keybinds):
    with open(SAVE_FILE,'w') as f:
        json.dump(keybinds,f)


def load():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            data = json.load(f)
            keybinds.open.value = data.get('open',openvalue)
            keybinds.closed.value = data.get('closed',closedvalue)
            keybinds.left.value = data.get('left',leftvalue)
            keybinds.right.value = data.get('left',rightvalue)
            keybinds.finger.value = data.get('finger',fingervalue)
            keybinds.peace.value = data.get('peace',peacevalue)

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
