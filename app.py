import webview
from flask import Flask, render_template, request, jsonify
import threading
import os
import json 

SAVE_FILE = 'saved_data.json'

keybinds = { 'open' : 'Scroll Y',
            'closed' : 'Scroll X',
            'left' : 'Ctrl Z',
            'right' : 'Ctrl X',
            'finger' : 'B',
            'peace' : 'E',
            }

def save(keybinds):
    with open(SAVE_FILE,'w') as f:
        json.dump(keybinds,f)


def load():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE) as f:
            data = json.load(f)
            data.get()

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/load')
def load_route():
    pass

@app.route('/api/buttonclick',methods=['POST'])
def button_click():
    pass

# can create object then update with keybinds with for loop then save