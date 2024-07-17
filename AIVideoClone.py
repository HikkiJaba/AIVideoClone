import os
import cv2
import numpy as np
import pyttsx3
from flask import Flask, request, render_template, jsonify, send_from_directory
import imghdr
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'

def text_to_speech(text, gender='neutral'):
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    
    if gender == 'male':
        engine.setProperty('voice', voices[0].id)
    elif gender == 'female':
        engine.setProperty('voice', voices[1].id)
    
    file_name = f"output_{text[:10]}.mp3"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    engine.save_to_file(text, file_path)
    engine.runAndWait()

    return file_name

def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted {filename}")
    else:
        print(f"File {filename} not found")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    data = request.json
    text = data.get('text')
    gender = data.get('gender', 'neutral')

    file_name = text_to_speech(text, gender)

    return jsonify({'status': 'success', 'file': file_name})

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file part"
        
        file = request.files["file"]
        
        if file.filename == "":
            return "No selected file"
        
        file_stream = file.read()
        image_type = imghdr.what(None, file_stream)
        
        if not image_type:
            return "Uploaded file is not an image"
        
        nparr = np.frombuffer(file_stream, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None or image.size == 0:
            return "Failed to load or decode image file"
        
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

        faces = face_cascade.detectMultiScale(image_gray)

        if len(faces) > 0:
            result_text = "Лицо обнаружено на изображении."
        else:
            result_text = "Лиц не обнаружено на изображении."

        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template("result.html", result=result_text, image=img_base64)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
