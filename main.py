import torch
from TTS.api import TTS

# Определите устройство
device = "cuda" if torch.cuda.is_available() else "cpu"

# Список доступных моделей
print(TTS().list_models())

# Инициализация TTS
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# Синтез речи
# Использование модели клониронyвания голоса
wav = tts.tts(text="Дбрый день! Как ваши дела ?", speaker_wav="audio.wav", language="ru")

# Сохранение синтезированного аудио в файл
tts.tts_to_file(text="Дбрый день! Как ваши дела ?", speaker_wav="audio.wav", language="ru", file_path="output.wav")
