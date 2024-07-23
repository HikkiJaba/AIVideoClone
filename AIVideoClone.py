import os
import cv2
import numpy as np
import torch
from flask import Flask, request, render_template, jsonify, send_from_directory
import base64
from TTS.api import TTS
import librosa
import parselmouth
from parselmouth.praat import call
import dlib
import imghdr
import soundfile as sf
import moviepy.editor as mpe

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
app.config['SPEAKER_FOLDER'] = 'speakers'

# Определите устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Инициализация TTS для русской речи
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Инициализация dlib для обнаружения лиц и предсказания лицевых ориентиров
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Словарь сопоставления фонем с формами рта для русского языка
phoneme_to_mouth_shape = {
    'а': 'open',
    'е': 'semi_open',
    'и': 'closed',
    'о': 'semi_open',
    'у': 'closed',
    'п': 'closed',
    'б': 'closed',
    'м': 'closed',
    'ф': 'semi_open',
    'в': 'semi_open',
    'т': 'closed',
    'д': 'closed',
    'н': 'closed',
    'с': 'semi_open',
    'з': 'semi_open',
    'л': 'semi_open',
    'р': 'open',
    'ы': 'closed',
    'й': 'closed',
    'ц': 'semi_open',
    'ч': 'semi_open',
    'ш': 'semi_open',
    'щ': 'semi_open',
    'ж': 'semi_open',
    'ю': 'semi_open',
    'я': 'semi_open',
    'х': 'semi_open',
    'ъ': 'closed',
    'ь': 'closed'
}

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

def get_phoneme_timestamps(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    temp_audio_path = "temp_audio.wav"
    sf.write(temp_audio_path, y, sr)
    snd = parselmouth.Sound(temp_audio_path)
    tg = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
    num_intervals = call(tg, "Get number of intervals", 1)
    phoneme_timestamps = []

    for i in range(1, num_intervals + 1):
        start_time = call(tg, "Get start time of interval", 1, i)
        end_time = call(tg, "Get end time of interval", 1, i)
        label = call(tg, "Get label of interval", 1, i)
        phoneme_timestamps.append((start_time, end_time, label))
    
    return phoneme_timestamps

def get_mouth_shape(phoneme):
    return phoneme_to_mouth_shape.get(phoneme, 'neutral')

def apply_mouth_shape(img, landmarks, mouth_shape):
    print(f"Applying mouth shape: {mouth_shape}")
    mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in range(48, 61)])
    mouth_shape_map = {
        'open': 5,
        'semi_open': 2,
        'closed': -2,
        'neutral': 0
    }
    shift = mouth_shape_map.get(mouth_shape, 0)

    if shift != 0:
        mouth_points[:, 1] += shift

    mask = np.zeros_like(img, dtype=np.uint8)
    hull = cv2.convexHull(mouth_points)
    cv2.fillConvexPoly(mask, hull, (255, 255, 255))

    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    img_with_mouth = img.copy()
    img_with_mouth[mask_gray > 0] = (img_with_mouth[mask_gray > 0] * 0.7 + mask[mask_gray > 0] * 0.3).astype(np.uint8)
    
    return img_with_mouth

def create_video_from_frames(frames, video_path, fps=30):
    if not frames:
        raise ValueError("No frames to create video")
        
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    
    for frame in frames:
        video.write(frame)
    
    video.release()

def animate_mouth(image_path, phoneme_timestamps, fps=30):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image could not be loaded.")
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    if len(faces) == 0:
        raise ValueError("No faces detected in the image.")
    
    frames = []
    for face in faces:
        landmarks = predictor(gray, face)
        for start_time, end_time, phoneme in phoneme_timestamps:
            mouth_shape = get_mouth_shape(phoneme)
            print(f"Processing phoneme: {phoneme}, mouth shape: {mouth_shape}")
            duration = end_time - start_time
            num_frames = int(duration * fps)
            for i in range(num_frames):
                t = start_time + (i / num_frames) * duration
                frame = img.copy()
                frame = apply_mouth_shape(frame, landmarks, mouth_shape)
                frames.append(frame)
    
    if not frames:
        frames.append(img)

    video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'animated_video.mp4')
    create_video_from_frames(frames, video_path, fps)
    return video_path

def combine_audio_video(video_path, audio_path, output_path):
    video = mpe.VideoFileClip(video_path)
    audio = mpe.AudioFileClip(audio_path)
    final_clip = video.set_audio(audio)
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

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
        phoneme_timestamps = get_phoneme_timestamps(file_path)
        print(f"Phoneme timestamps: {phoneme_timestamps}")
        
        image_file = request.files.get('image_file')
        if image_file:
            image_path = save_file(image_file, app.config['UPLOAD_FOLDER'])
            print(f"Image file saved: {image_path}")
            video_path = animate_mouth(image_path, phoneme_timestamps)
            print(f"Video created: {video_path}")
            
            final_video_path = os.path.join(app.config['UPLOAD_FOLDER'], 'final_animated_video.mp4')
            combine_audio_video(video_path, file_path, final_video_path)
            print(f"Final video created: {final_video_path}")
            
            return render_template('result.html', result='Анимация создана', image=None, video=os.path.basename(final_video_path))
        else:
            return render_template('result.html', result='Только аудио создано', image=None, video=None)
    else:
        return jsonify({'status': 'error', 'message': 'Failed to synthesize speech'}), 500

@app.route('/audio/<filename>')
def get_audio(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/video/<filename>')
def get_video(filename):
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
