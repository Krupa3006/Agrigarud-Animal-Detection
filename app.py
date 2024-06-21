import os
from flask import Flask, render_template
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run_script', methods=['POST'])
def run_script():
    script_path = os.path.join(os.getcwd(), 'main.py')  # Assuming main.py is in the same directory as this script
    subprocess.Popen(["python", script_path])
    message = "Animal detection started successfully!"
    return render_template('index.html', message=message)

if __name__ == '__main__':
    app.run(debug=True)