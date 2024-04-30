import argparse
import os
import numpy as np
import museval
import torch
import soundfile as sf
from models.unet import UNet
from data.preprocess import to_mag
import librosa
import tqdm
from inference import separate, save_audio
from config import WINDOW_SIZE, HOP_LENGTH, SAMPLING_RATE, DEVICE, SEGMENT_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description='Music Source Separation Evaluation')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input mixture files')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--output', type=str, default='./eval_results.txt', help='Output file to save evaluation results')
    # parser.add_argument('--segment_size', type=int, default=2048, help='Segment size')
    return parser.parse_args()

def evaluate(input_dir, model_path, segment_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    model = UNet(in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sdrs = []
    isrs = []
    sirs = []
    sars = []

    song_dirs = [os.path.join(input_dir, d) for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]
    for song_dir in tqdm.tqdm(song_dirs, desc="Evaluating songs", unit="song"):
    # for song_dir in os.listdir(input_dir):
        # song_path = os.path.join(input_dir, song_dir)
        # if not os.path.isdir(song_path):
            # continue

        # Load source references in their original sr and channel number
        target_sources = []
        for source in ['bass', 'drums', 'other', 'vocals']:
            # target_path = os.path.join(song_path, f'{source}.wav')
            target_path = os.path.join(song_dir, f'{source}.wav')
            # print("path", target_path)
            target_audio, _ = librosa.load(target_path, sr=SAMPLING_RATE, mono=False)
            target_sources.append(target_audio.T)
        target_sources = np.stack(target_sources)

        # Predict using mixture
        # mixture_path = os.path.join(song_path, 'mixture.wav')
        mixture_path = os.path.join(song_dir, 'mixture.wav')
        pred_sources = predict_song(mixture_path, model, SEGMENT_SIZE)
        pred_sources = np.stack([pred_sources[key] for key in ['bass', 'drums', 'other', 'vocals']])

        pred_sources= np.stack((pred_sources, pred_sources), axis=-1)

        # print("target", target_sources.shape)
        # print("pred", pred_sources.shape)

        # Evaluate
        SDR, ISR, SIR, SAR, _ = museval.metrics.bss_eval(target_sources, pred_sources)
        # print("SDR", SDR.shape)

        avg_sdr = np.nanmean(SDR, axis=1)
        avg_isr = np.nanmean(ISR, axis=1)
        avg_sir = np.nanmean(SIR, axis=1)
        avg_sar = np.nanmean(SAR, axis=1)

        # print("SDR", avg_sdr.shape)

        sdrs.append(avg_sdr)
        isrs.append(avg_isr)
        sirs.append(avg_sir)
        sars.append(avg_sar)

    avg_sdr = np.nanmean(sdrs)
    avg_isr = np.nanmean(isrs)
    avg_sir = np.nanmean(sirs)
    avg_sar = np.nanmean(sars)

    return avg_sdr, avg_isr, avg_sir, avg_sar

def load_audio(file):
    # print("file", file)
    if file.endswith('.wav'):
        wav, _ = librosa.load(file, sr=SAMPLING_RATE)
        spec = librosa.stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
        mag = np.abs(spec)
        return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(mag).unsqueeze(0)
    elif file.endswith('.npz'):
        data = np.load(file)
        return data['mix']
    else:
        raise ValueError(f'Unsupported file format: {file}')

def predict_song(mixture_path, model, segment_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  

    # mix_wav = load_audio(mixture_path)

    mix_spec, mix_mag = load_audio(mixture_path)
    mix_spec = mix_spec.to(device)
    mix_mag = mix_mag.to(device)
    output = separate(model, mix_spec, mix_mag, SEGMENT_SIZE)

    # mix_spec = librosa.stft(mix_wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
    # mix_mag = np.abs(mix_spec)
    # mix_phase = np.angle(mix_spec)

    # mix_mag_tensor = torch.from_numpy(mix_mag).unsqueeze(0).to(device)

    # output = separate(model, mix_spec, mix_mag_tensor, SEGMENT_SIZE)

    mask = output.squeeze(0)
    mix_spec_real = mix_spec.real
    mix_spec_imag = mix_spec.imag
    mix_spec_complex = torch.stack([mix_spec_real, mix_spec_imag], dim=-1)

    separated = mask.unsqueeze(3) * mix_spec_complex.unsqueeze(0)
    separated = separated.squeeze(0)
    # print("separated", separated.shape)

    window = torch.hann_window(WINDOW_SIZE, device=device)

    output = output.cpu().numpy()

    pred_sources = {}
    for i, source in enumerate(['bass', 'drums', 'other', 'vocals']):
        source_wav = torch.istft(separated[i], WINDOW_SIZE, window=window, hop_length=HOP_LENGTH, length=output.shape[-1]*HOP_LENGTH)
        source_wav = source_wav.cpu().numpy()
        pred_sources[source] = source_wav

    return pred_sources

def main():
    args = parse_args()

    avg_sdr, avg_isr, avg_sir, avg_sar = evaluate(args.input_dir, args.model, SEGMENT_SIZE)

    print(f'Average SDR: {avg_sdr:.4f}')
    print(f'Average ISR: {avg_isr:.4f}')
    print(f'Average SIR: {avg_sir:.4f}')
    print(f'Average SAR: {avg_sar:.4f}')

    with open(args.output, 'w') as f:
        f.write(f'Average SDR: {avg_sdr:.4f}\n')
        f.write(f'Average ISR: {avg_isr:.4f}\n')
        f.write(f'Average SIR: {avg_sir:.4f}\n')
        f.write(f'Average SAR: {avg_sar:.4f}\n')

    print("Evaluation complete. Results saved to:", args.output)

if __name__ == '__main__':
    main()