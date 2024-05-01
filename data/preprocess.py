import os
import librosa
import numpy as np
from config import WINDOW_SIZE, HOP_LENGTH, SAMPLING_RATE, DEVICE
import torch
import torchaudio

def to_mag(file):
    # import pdb;
    # pdb.set_trace()
    wav, _ = librosa.load(file, sr=SAMPLING_RATE, mono=True)    # (3776000,)
    wav_tensor = torch.from_numpy(wav).to(DEVICE)  # [3776000]
    # wav, b = torchaudio.load(file)
    # waveform = torch.mean(wav, dim=0, keepdim=True)
    # print("wave", wav.shape)
    window = torch.hann_window(WINDOW_SIZE, device=DEVICE)
    # spectrogram = librosa.stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)  [1024, 7375, 2]
    spectrogram = torch.stft(wav_tensor, n_fft=WINDOW_SIZE, window=window) # [1024, 7390, 2]
    # print("spectrogram", spectrogram.shape)
    # mag, _ = librosa.magphase(spectrogram)
    mag = spectrogram.pow(2).sum(-1).sqrt()  # [1024,7390]
    # print("mag", mag.shape)
    mag = mag.cpu().numpy().astype(np.float32)
    return mag
    # return mag

def save_to_npz(base, sub, sample):
    mix = to_mag(f'{base}/{sample}/mixture.wav')
    bass = to_mag(f'{base}/{sample}/bass.wav')
    drums = to_mag(f'{base}/{sample}/drums.wav')
    other = to_mag(f'{base}/{sample}/other.wav')
    vocals = to_mag(f'{base}/{sample}/vocals.wav')
    
    # mix_max = mix.max()
    # mix_min = mix.min()
    # bass_max = bass.max()
    # bass_min = bass.min()
    # drums_max = drums.max()
    # drums_min = drums.min()
    # other_max = other.max()
    # other_min = other.min()
    # vocals_max = vocals.max()
    # vocals_min = vocals.min()
    
    # mix_norm = (mix - mix_min) / (mix_max - mix_min)
    # bass_norm = (bass - bass_min) / (bass_max - bass_min)
    # drums_norm = (drums - drums_min) / (drums_max - drums_min)
    # other_norm = (other - other_min) / (other_max - other_min)
    # vocals_norm = (vocals - vocals_min) / (vocals_max - vocals_min)
    
    if not os.path.exists(f'../musdb18_npz/{sub}'):
        os.makedirs(f'../musdb18_npz/{sub}')
    
    print(f"Saving {sample}")
    # np.savez_compressed(f'../musdb18_npz/{sub}/{sample}.npz',
    #                     mix=mix_norm, bass=bass_norm, drums=drums_norm,
    #                     other=other_norm, vocals=vocals_norm)

    np.savez_compressed(f'../musdb18_npz/{sub}/{sample}.npz',
                        mix=mix, bass=bass, drums=drums,
                        other=other, vocals=vocals)

# def preprocess_data(data_dir):
#     if not os.path.exists('../musdb18_npz'):
#         os.makedirs('../musdb18_npz')
    
#     for sample in os.listdir(data_dir):
#         save_to_npz(data_dir, 'train', sample)

def preprocess_data(data_dir, is_train=True):
    if not os.path.exists('../musdb18_npz'):
        os.makedirs('../musdb18_npz')
    
    sub_dir = 'train' if is_train else 'test'
    
    for sample in os.listdir(data_dir):
        save_to_npz(data_dir, sub_dir, sample)

    target_dir = '../musdb18_npz'
    if is_train:
        path = os.path.join(target_dir, 'train')     
    else:
        path = os.path.join(target_dir, 'test') 
    return path     