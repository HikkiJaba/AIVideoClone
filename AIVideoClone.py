import os
import time
from flask import Flask, request, render_template, send_from_directory
import cv2
import numpy as np
import mediapipe as mp
import subprocess

# Функция для извлечения ключевых точек лица
def get_face_landmarks(image_path):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    img = cv2.imread(image_path)
    with mp_face_mesh.FaceMesh(max_num_faces=1) as face_mesh:
        results = face_mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        return None

# Функция для синхронизации губ с аудио с помощью Wav2Lip
def sync_lips_with_audio(image_path, audio_path, output_video_path):
    # Запускаем Wav2Lip с помощью команды
    subprocess.run([
        'python', 'inference.py',
        '--checkpoint_path', 'checkpoints/wav2lip_gan.pth',
        '--face', image_path,
        '--audio', audio_path,
        '--outfile', output_video_path
    ])

# Функция для создания финального видео с анимацией
def create_video_with_animation(image_path, video_path, output_video_path):
    img = cv2.imread(image_path)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка открытия видеофайла.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24.0, (img.shape[1], img.shape[0]))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Наложите анимацию на изображение
        frame_img = cv2.addWeighted(img, 0.5, frame, 0.5, 0)
        out.write(frame_img)

    cap.release()
    out.release()
    print(f'Видео сохранено как {output_video_path}')

# Пример использования
def main():
    image_path = 'path_to_your_image.jpg'
    audio_path = 'path_to_your_audio.wav'
    wav2lip_output = 'output_video.mp4'
    final_output = 'final_output_video.avi'

    # Извлечение ключевых точек лица
    landmarks = get_face_landmarks(image_path)
    if landmarks:
        print("Ключевые точки лица извлечены.")
    
    # Синхронизация губ с аудио
    print("Синхронизация губ с аудио...")
    sync_lips_with_audio(image_path, audio_path, wav2lip_output)
    
    # Создание финального видео
    print("Создание финального видео...")
    create_video_with_animation(image_path, wav2lip_output, final_output)

if __name__ == "__main__":
    main()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
FINISH_VIDEO_PATH = 'static/finish.mp4'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/synthesize', methods=['POST'])
def synthesize():
    # Принимаем данные (но ничего с ними не делаем)
    data = request.form.get('data')
    
    # Ждем 3 секунды
    time.sleep(6)

    # Путь к видео
    video_url = f"/file/finish.mp4"
    return render_template('result.html', result='Анимация создана', video=video_url)

@app.route('/file/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
