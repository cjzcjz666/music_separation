import librosa
import numpy as np

def load_audio(file, sr):
    wav, _ = librosa.load(file, sr=sr)
    return wav

def save_audio(wav, file, sr):
    librosa.output.write_wav(file, wav, sr=sr)