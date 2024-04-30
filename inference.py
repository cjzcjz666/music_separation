import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from models.unet import UNet
from data.preprocess import to_mag
from config import WINDOW_SIZE, HOP_LENGTH, SAMPLING_RATE, SEGMENT_SIZE

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
        wav, _ = librosa.load(file, sr=SAMPLING_RATE)
        spec = librosa.stft(wav, n_fft=WINDOW_SIZE, hop_length=HOP_LENGTH)
        mag = np.abs(spec)
        return torch.from_numpy(spec).unsqueeze(0), torch.from_numpy(mag).unsqueeze(0)
    elif file.endswith('.npz'):
        data = np.load(file)
        return data['mix']
    else:
        raise ValueError(f'Unsupported file format: {file}')

def separate(model, mix_spec, mix_mag, segment_size):
    # Pad the input to ensure it's a multiple of segment_size
    mix_length = mix_mag.size(-1)
    padding = segment_size - mix_length % segment_size
    mix_mag = torch.nn.functional.pad(mix_mag, (0, padding))
    # print("mix_mag.shape:", mix_mag.shape)
    # Split the input into segments
    # mix_segments = mix_mag.unfold(2, segment_size, segment_size).permute(2, 0, 1, 3)
    mix_segments = mix_mag.unfold(-1, segment_size, segment_size).permute(2, 0, 1, 3)
    # print("mix_segments", mix_segments.shape)
    
    # Process each segment
    output_segments = []
    for mix_segment in mix_segments:
        with torch.no_grad():
            mix_segment = mix_segment.unsqueeze(1)
            # print("mix_segment.shape:", mix_segment.shape)
            output_segment = model(mix_segment)
        output_segments.append(output_segment)
    
    # Reconstruct the output
    output = torch.cat(output_segments, dim=-1)
    output = output[..., :mix_length]  # Remove padding
    
    return output

def save_audio(output, mix_spec, output_dir):
    sources = ['bass', 'drums', 'other', 'vocals']
    mix_phase = torch.angle(mix_spec)

    mask = output.squeeze(0)
    # print("mask", mask.shape)
    # print("output", output.shape)
    # print("mix_phase", mix_phase.shape)
    # print("mix_spec", mix_spec.shape)

    mix_spec_real = mix_spec.real
    mix_spec_imag = mix_spec.imag
    mix_spec_complex = torch.stack([mix_spec_real, mix_spec_imag], dim=-1)
    # print("mix_spec_complex", mix_spec_complex.shape)
    
    separated = mask.unsqueeze(3) * mix_spec_complex.unsqueeze(0)
    separated = separated.squeeze(0)
    # print("separated", separated.shape)

    window = torch.hann_window(WINDOW_SIZE, device='cpu')

    for i, source in enumerate(sources):
        # print("len", output.shape[-1]*HOP_LENGTH)
        source_wav = torch.istft(separated[i], WINDOW_SIZE, window=window, hop_length=HOP_LENGTH, length=output.shape[-1]*HOP_LENGTH)
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
    model.eval()
    
    if os.path.isdir(args.input):
        files = [os.path.join(args.input, f) for f in os.listdir(args.input)]
    else:
        files = [args.input]
    
    os.makedirs(args.output, exist_ok=True)
    
    for file in files:
        mix_spec, mix_mag = load_audio(file)
        mix_spec = mix_spec.to(DEVICE)
        mix_mag = mix_mag.to(DEVICE)
        output = separate(model, mix_spec, mix_mag, SEGMENT_SIZE)
        save_audio(output, mix_spec, args.output)

if __name__ == '__main__':
    args = parse_args()
    main(args)