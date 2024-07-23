import librosa
import parselmouth
from parselmouth.praat import call

def get_phoneme_timestamps(audio_path):
    # Загружаем аудио
    y, sr = librosa.load(audio_path, sr=None)
    
    # Сохраняем аудио во временный файл (если требуется)
    temp_audio_path = "temp_audio.wav"
    librosa.output.write_wav(temp_audio_path, y, sr)
    
    # Загружаем аудио в parselmouth
    snd = parselmouth.Sound(temp_audio_path)
    
    # Используем Praat для получения сегментации речи
    tg = call(snd, "To TextGrid (silences)", 100, 0, -25, 0.1, 0.1, "silent", "sounding")
    
    # Извлекаем временные метки фонем
    intervals = call(tg, "List?", "phoneme")
    phoneme_timestamps = []
    
    for i in range(1, len(intervals)):
        interval = call(tg, "Get interval", 1, i)
        start_time = call(interval, "Get start time")
        end_time = call(interval, "Get end time")
        label = call(interval, "Get label")
        phoneme_timestamps.append((start_time, end_time, label))
    
    return phoneme_timestamps

audio_path = "path/to/audio.wav"
phoneme_timestamps = get_phoneme_timestamps(audio_path)
print(phoneme_timestamps)


