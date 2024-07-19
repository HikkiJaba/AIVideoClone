import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
import base64
from TTS.api import TTS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SPEAKER_FOLDER'] = 'speakers'

# Определите устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

def save_file(file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    print(f"File saved to: {file_path}")  # Отладочное сообщение
    return file_path

def text_to_speech(text, speaker_wav=None, language='ru'):
    file_name = f"output_{text[:10]}.wav"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    
    try:
        if speaker_wav:
            if not os.path.exists(speaker_wav):
                raise FileNotFoundError(f"Speaker file not found: {speaker_wav}")
            wav = tts.tts(text=text, speaker_wav=speaker_wav, language=language)
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        else:
            tts.tts_to_file(text=text, language=language, file_path=file_path)
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return None
    
    return file_path

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form.get('text')
    gender = request.form.get('gender', 'neutral')
    speaker_file = request.files.get('speaker_file')
    
    if speaker_file:
        speaker_path = save_file(speaker_file, app.config['SPEAKER_FOLDER'])
    else:
        speaker_path = None

    file_path = text_to_speech(text, speaker_wav=speaker_path, language='ru')

    if file_path:
        return jsonify({'status': 'success', 'file': file_path})
    else:
        return jsonify({'status': 'error', 'message': 'Failed to synthesize speech'}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
