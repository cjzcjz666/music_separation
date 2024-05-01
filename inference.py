import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from models.unet import UNet
from data.preprocess import to_mag
from config import WINDOW_SIZE, HOP_LENGTH, SAMPLING_RATE, SEGMENT_SIZE, DEVICE
from utils.pad import padding

def parse_args():
    parser = argparse.ArgumentParser(description='Music Source Separation Inference')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    parser.add_argument('--input', type=str, required=True, help='Input audio file or directory')
    parser.add_argument('--output', type=str, default='./output', help='Output directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    return parser.parse_args()

def load_audio(file):
    print("file", file)
    if file.endswith('.wav'):
        # wav, _ = librosa.load(file, sr=SAMPLING_RATE)
        wav, _ = librosa.load(file, sr=SAMPLING_RATE, mono=True)
        window = torch.hann_window(WINDOW_SIZE, device=DEVICE)
        wav_tensor = torch.from_numpy(wav).to(DEVICE)
        length = wav_tensor.size(-1)
        spectrogram = torch.stft(wav_tensor, n_fft=WINDOW_SIZE, window=window)
        spec = spectrogram.pow(2).sum(-1).sqrt()
        # spec = librosa.stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
        # mag = np.abs(spec)
        spec = spec.cpu().numpy().astype(np.float32)
        return torch.from_numpy(spec), spectrogram, length
    elif file.endswith('.npz'):
        data = np.load(file)
        return data['mix']
    else:
        raise ValueError(f'Unsupported file format: {file}')

def separate(model, mix_spec):
    model.eval()

    with torch.no_grad():
        mix_padded, (left, right) = padding(mix_spec, 64)    # mix_spec [1024, 8645]   mix_padding [1024,8704]
        right = mix_padded.size(-1) - right
        
        # Add batch dimension and channel dimension
        input_new = mix_padded.unsqueeze(0).unsqueeze(0)  # [1,1,1024,8704]
        
        # Process the padded input
        output = model(input_new)  # [1,4,1024,8704]
        
        # Remove padding
        output = output[..., left:right]  # [1,4,1024,8645]
    
        return output

def save_audio(output, mix_spec, spectrogram, length, output_dir):
    sources = ['bass', 'drums', 'other', 'vocals']
    mix_phase = torch.angle(mix_spec)

    mask = output.squeeze(0)

    separated = mask.unsqueeze(3).to(DEVICE) * spectrogram.unsqueeze(0)
    separated = separated.squeeze(0)
    # print("separated", separated.shape)
    window = torch.hann_window(WINDOW_SIZE, device=DEVICE)
    
    for i, source in enumerate(sources):
        # print("len", output.shape[-1]*HOP_LENGTH)
        # import pdb;
        # pdb.set_trace()
        source_wav = torch.istft(separated[i], WINDOW_SIZE, window=window, length=length)
        # source_spec = output[i].unsqueeze(-1) * torch.exp(1j * mix_phase)
        # source_wav = torch.istft(source_spec.squeeze(0), n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH, length=output.shape[-1]*HOP_LENGTH)
        source_wav = source_wav.cpu().numpy()
        sf.write(os.path.join(output_dir, f'{source}.wav'), source_wav.T, SAMPLING_RATE)

def main(args):
    if args.gpu and torch.cuda.is_available():
        DEVICE = torch.device('cuda')
    else:
        DEVICE = torch.device('cpu')
        
    model = UNet(in_channels=1, out_channels=4)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    else:
        files = [args.input]
    
    os.makedirs(args.output, exist_ok=True)
    
    for file in files:
        mix_spec, spectrogram, length = load_audio(file)
        mix_spec = mix_spec.to(DEVICE)
        # mix_mag = mix_mag.to(DEVICE)
        output = separate(model, mix_spec)
        save_audio(output, mix_spec, spectrogram, length, args.output)

if __name__ == '__main__':
    args = parse_args()
    main(args)
