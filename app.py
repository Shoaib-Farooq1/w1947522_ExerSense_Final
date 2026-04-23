import sys

from flask import Flask, render_template, jsonify
import json
import subprocess
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status')
def status():
    try:
        with open('state.json', 'r') as f:
            data = json.load(f)
        return jsonify(data)
    except:
        return jsonify({"score": 0, "rom": 0, "status": "waiting"})

@app.route('/start')
def start():
    subprocess.Popen(
        [sys.executable, 'exersense_realtime.py'],
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    return jsonify({'ok': True})

@app.route('/set')
def working_set():
    with open('start_set.flag', 'w') as f:
        f.write('start')
    return jsonify({'ok': True})

@app.route('/stop')
def stop():
    with open('stop_set.flag', 'w') as f:
        f.write('stop')
    return jsonify({'ok': True})

@app.route('/reset')
def reset():
    with open('state.json', 'w') as f:
        json.dump({"score": 0, "rom": 0, "status": "waiting"}, f)
    return jsonify({'ok': True})

if __name__ == '__main__':
    app.run(debug=True)