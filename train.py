import argparse
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

import data
from u_net import UNet, padding

N_PART = 4
N_FFT = 2047
SAMPLING_RATE = 22050


def train(model, data_loader, optimizer, device, epoch, tb_writer):
    model.train()
    total_loss = 0
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")
    
    for x, t in progress_bar:
        batch_size = x.size(0)
        x, t = x.to(device), t.to(device)

        # print("x", x.shape)
        # print("t", t.shape)

        y = model(x) * x.unsqueeze(1)
        # print("y", y.shape)
        loss = F.l1_loss(y, t, reduction='sum') / batch_size
        total_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    tb_writer.add_scalar('train/loss', total_loss / len(data_loader), epoch)


def test(model, test_data, device, epoch, tb_writer):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        window = torch.hann_window(N_FFT, device=device)
        for sound in test_data:
            sound = sound.to(device)
            sound_stft = torch.stft(sound, N_FFT, window=window)
            sound_spec = sound_stft.pow(2).sum(-1).sqrt()
            x, t = sound_spec[0], sound_spec[1:]

            # print("x", x.shape)

            x_padded, (left, right) = padding(x)

            # print("x_padded", x_padded.shape)

            right = x_padded.size(1) - right

            input_new = x_padded.unsqueeze(0)
            # print("input", input_new.shape)

            mask = model(x_padded.unsqueeze(0)).squeeze(0)[:, :, left:right]
            y = mask * x.unsqueeze(0)
            loss = F.l1_loss(y, t, reduction='sum')
            total_loss += loss.item()

    # TODO: Also evaluate separation performance
    tb_writer.add_scalar('test/loss', total_loss / len(test_data), epoch)


def main():
    parser = argparse.ArgumentParser(
        description='Train U-Net with MUSDB18 dataset.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset',
                        help='Path of dataset which converted to .wav format '
                             'by `convert_to_wav.py`.',
                        type=str, metavar='PATH', required=True)
    parser.add_argument('--batch-size', '-b',
                        help='Batch size',
                        type=int, default=64)
    parser.add_argument('--epochs', '-e',
                        help='Number of epochs',
                        type=int, default=500)
    parser.add_argument('--output-interval',
                        help='Save model per N epochs',
                        type=int, metavar='N', default=1)
    parser.add_argument('--eval-interval',
                        help='Evaluate and save model per N epochs',
                        type=int, metavar='N', default=25)
    parser.add_argument('--gpu', '-g',
                        help='GPU id (Negative number indicates CPU)',
                        type=int, nargs='+', metavar='ID', default=[0])
    parser.add_argument('--learning-rate', '-l',
                        help='Learning rate',
                        type=float, metavar='LR', default=2e-3)
    parser.add_argument('--no-cuda',
                        help='Do not use GPU',
                        action='store_true')
    parser.add_argument('--output',
                        help='Save model to PATH',
                        type=str, metavar='PATH', default='./models')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    if_use_cuda = torch.cuda.is_available() and not args.no_cuda
    if if_use_cuda:
        torch.backends.cudnn.benchmark = True
    device = torch.device(f'cuda:{args.gpu[0]}' if if_use_cuda else 'cpu')

    model = UNet(N_PART)
    if not args.no_cuda and len(args.gpu) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Dataloader
    train_data, test_data =\
        data.read_data(args.dataset, N_FFT, 512, SAMPLING_RATE)
    train_dataset = data.RandomCropDataset(train_data, 256)
    print("train_dataset", len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, args.batch_size, shuffle=True,
        num_workers=2, pin_memory=False)

    # Tensorboard
    tb_writer = SummaryWriter()

    for epoch in range(1, args.epochs + 1):
        train(model, train_loader, optimizer, device, epoch, tb_writer)
        if epoch % args.output_interval == 0:
            # Save the model
            test(model, test_data, device, epoch, tb_writer)
            model.cpu()
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
            path = os.path.join(args.output, f'model-{epoch}.pth')
            torch.save(state_dict, path)
            model.to(device)

    tb_writer.close()


if __name__ == '__main__':
    main()
