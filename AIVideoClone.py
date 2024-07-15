import cv2
import numpy as np
from flask import Flask, request, render_template
import imghdr
import base64

app = Flask(__name__)

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

        # Конвертируем изображение в base64 строку
        _, img_encoded = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')

        return render_template("result.html", result=result_text, image=img_base64)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
