import os
import requests
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
import base64
from TTS.api import TTS
import librosa
import torch
import parselmouth
from parselmouth.praat import call
import dlib
import soundfile as sf
import moviepy.editor as mpe

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SPEAKER_FOLDER'] = 'speakers'
GOOEY_API_KEY = "sk-jHiwd5KGESRxsGcDH4GCs19kGHs4Fq9H40WOpz4xyW4vzKiN"

@app.route('/')
def index():
    return render_template('index.html')

# Определите устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация TTS для русской речи
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Инициализация dlib для обнаружения лиц и предсказания лицевых ориентиров
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def save_file(file, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    file_path = os.path.join(folder, file.filename)
    file.save(file_path)
    print(f"File saved to: {file_path}")
    return file_path

def text_to_speech(text, speaker_wav=None, language='ru'):
    file_name = f"output_{text[:10]}.wav"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
    
    try:
        if speaker_wav:
            if not os.path.exists(speaker_wav):
                raise FileNotFoundError(f"Speaker file not found: {speaker_wav}")
            tts.tts_to_file(text=text, speaker_wav=speaker_wav, language=language, file_path=file_path)
        else:
            tts.tts_to_file(text=text, language=language, file_path=file_path)
    except Exception as e:
        print(f"Error during TTS synthesis: {e}")
        return None
    
    return file_path

def combine_audio_video(video_path, audio_path, output_path):
    video = mpe.VideoFileClip(video_path)
    audio = mpe.AudioFileClip(audio_path)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

def animate_mouth_with_api(image_url, audio_url):
    url = "https://api.gooey.ai/v2/Lipsync/"
    headers = {
        "Authorization": f"Bearer {GOOEY_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "input_face": image_url,
        "input_audio": audio_url,
        "selected_model": "Wav2Lip",
        "settings": {
            "retention_policy": "keep"
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        return result['output']['output_video']
    else:
        raise Exception(f"Failed to animate mouth: {response.status_code}, {response.text}")

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.form.get('text')
    speaker_file = request.files.get('speaker_file')
    
    if speaker_file:
        speaker_path = save_file(speaker_file, app.config['SPEAKER_FOLDER'])
    else:
        speaker_path = None

    print(f"Synthesizing text: {text}")
    file_path = text_to_speech(text, speaker_wav=speaker_path, language='ru')

    if file_path:
        print(f"Audio file created: {file_path}")
        
        image_file = request.files.get('image_file')
        if image_file:
            image_path = save_file(image_file, app.config['UPLOAD_FOLDER'])
            print(f"Image file saved: {image_path}")

            # Загрузка изображений и аудиофайлов на сервер для получения URL
            with open(image_path, "rb") as img_file:
                image_data = base64.b64encode(img_file.read()).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{image_data}"

            with open(file_path, "rb") as audio_file:
                audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
            audio_url = f"data:audio/wav;base64,{audio_data}"
            
            try:
                output_video_url = animate_mouth_with_api(image_url, audio_url)
                print(f"Video created: {output_video_url}")
                
                return render_template('result.html', result='Анимация создана', image=None, video=output_video_url)
            except Exception as e:
                print(f"Error: {e}")
                return render_template('result.html', result=f'Ошибка: {e}', image=None, video=None)
        else:
            return render_template('result.html', result='Только аудио синтезировано, изображение отсутствует', image=None, video=None)
    else:
        return render_template('result.html', result='Ошибка синтеза аудио', image=None, video=None)

@app.route('/file/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
