import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from data.dataset import MusicDataset
from models.unet import UNet
from config import DEVICE
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import datetime
from utils.loss import SpectralLoss
from config import WINDOW_SIZE, HOP_LENGTH, SAMPLING_RATE, SEGMENT_SIZE

def parse_args():
    parser = argparse.ArgumentParser(description='Music Source Separation')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--gpu_ids', type=str, default='0', help='GPU IDs to use')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs')
    parser.add_argument('--early_stopping', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--dataset', type=str, default='../musdb18_wav/train', help='Dataset directory')
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log directory for TensorBoard')
    parser.add_argument('--model_dir', type=str, default='./model', help='Model directory')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--loss', type=str, default='l1', choices=['mse', 'l1'], help='Loss function')
    parser.add_argument('--l2_reg', type=float, default=0, help='L2 regularization weight')
    parser.add_argument('--use_cosine_annealing', action='store_true', help='Use cosine annealing learning rate scheduler')
    # parser.add_argument('--segment_size', type=int, default=2048, help='Segment size')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained model')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--use_spectral_loss', action='store_true', help='Use spectral loss in addition to the main loss')
    parser.add_argument('--spectral_loss_weight', type=float, default=0.1, help='Weight for the spectral loss')
    return parser.parse_args()

# def train(args, model, dataloader, loss_fn, spectral_loss_fn, optimizer, scheduler, writer):
def train(args, model, train_loader, test_loader, loss_fn, spectral_loss_fn, optimizer, scheduler, writer):
    best_loss = float('inf')
    early_stop_counter = 0

    for epoch in range(args.epochs):
        model.train()
        avg_loss = 0.0
        
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.epochs}", unit='batch') as pbar:
            for i, (x_batch, y_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                x_batch = x_batch.to(DEVICE)
                x_batch = x_batch.unsqueeze(1)
                y_batch = y_batch.to(DEVICE)
            
                # print("x_batch", x_batch.shape)
                # print("y_batch", y_batch.shape)

                y_pred = model(x_batch)

                # print("y_batch", y_batch.shape)
                # print("y_pred", y_pred.shape)

                loss = loss_fn(y_pred, y_batch)

                l2_reg_loss = 0
                for param in model.parameters():
                    l2_reg_loss += torch.norm(param)
                loss += args.l2_reg * l2_reg_loss

                if args.use_spectral_loss:
                    print("use spectral_loss")
                    spectral_loss = spectral_loss_fn(y_pred, y_batch)
                    loss += args.spectral_loss_weight * spectral_loss

                avg_loss += loss.item()
            
                loss.backward()
                optimizer.step()

                if args.use_cosine_annealing:
                    scheduler.step()

                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss /= len(train_loader)
        test_loss = test(model, test_loader, SEGMENT_SIZE, loss_fn)
        
        writer.add_scalar('Loss/train', avg_loss, epoch)
        print(f'Epoch {epoch+1}: Train Loss = {avg_loss}')

        writer.add_scalar('Loss/test', test_loss, epoch)
        print(f'Epoch {epoch+1}: Test Loss = {test_loss}')
        
        if test_loss < best_loss:
            best_loss = test_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pt'))
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stopping:
                print(f'Early stopping at epoch {epoch}')
                break
        
        torch.save(model.state_dict(), os.path.join(model_dir, 'latest_model.pt'))

# def test(model, test_loader, segment_size, loss_fn):
#     model.eval()
#     test_loss = 0.0
#     with torch.no_grad():
#         for mix, target in test_loader:
#             mix = mix.to(DEVICE)
#             # print("mix_old", mix.shape)
#             target = target.to(DEVICE)
            
#             # Pad the input to ensure it's a multiple of segment_size
#             mix_length = mix.size(-1)
#             padding = segment_size - mix_length % segment_size
#             mix = torch.nn.functional.pad(mix, (0, padding))
#             target = torch.nn.functional.pad(target, (0, padding))  # Pad the last dimension
            
#             # Split the input into segments
#             mix_segments = mix.unfold(2, segment_size, segment_size).permute(2, 0, 1, 3)
            
#             # Process each segment
#             output_segments = []
#             for mix_segment in mix_segments:
#                 output_segment = model(mix_segment.unsqueeze(1))
#                 output_segments.append(output_segment)
            
#             # Reconstruct the output
#             output = torch.cat(output_segments, dim=-1)  # Concatenate along the last dimension
#             output = output[..., :mix_length]  # Remove padding from the last dimension
            
#             # print("mix_new", mix.shape)
#             # print("output", output.shape)
#             # print("target", target.shape)
            
#             loss = loss_fn(output, target[..., :mix_length])  # Remove padding from target
#             test_loss += loss.item()
    
#     test_loss /= len(test_loader)
#     return test_loss

def test(model, test_loader, segment_size, loss_fn):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for mix, target in test_loader:
            mix = mix.to(DEVICE)
            target = target.to(DEVICE)
            # print("target", target.shape)
            # print("mix", mix.shape)

            # Pad the input to ensure it's a multiple of 64
            mix_padded, (left, right) = padding(mix, 64)

            # print("mix_padded", mix_padded.shape)
            target_padded, _ = padding(target, 64)
            right = mix_padded.size(-1) - right
            
            input_new = mix_padded.unsqueeze(0)
            # print("input", input_new.shape)

            # Process the padded input
            output = model(input_new)
            # print("output1", output.shape)
            output = output[..., left:right]

            # print("output2", output.shape)
            # print("target_padded", target_padded.shape)

            # a = target_padded[..., left:right]
            # print("a", a.shape)
            
            loss = loss_fn(output, target_padded[..., left:right])
            test_loss += loss.item()
    
    test_loss /= len(test_loader)
    return test_loss

def padding(signal, pad_multiple):
    """Apply padding to ensure that the number of time frames of `signal` is a multiple of `pad_multiple`.
    
    Args:
        signal (torch.Tensor): Signal to be padded.
        pad_multiple (int): Desired multiple of the padded signal length.
        
    Returns:
        Tuple[torch.Tensor, Tuple[int, int]]: Padded signal and the number of frames padded to the left and right sides, respectively.
    """
    n_frames = signal.size(-1)
    n_pad = (pad_multiple - n_frames % pad_multiple) % pad_multiple
    if n_pad:
        left = n_pad // 2
        right = n_pad - left
        return torch.nn.functional.pad(signal, (left, right)), (left, right)
    else:
        return signal, (0, 0)

if __name__ == '__main__':
    args = parse_args()
    
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(args.log_dir, current_time)
    model_dir = os.path.join(args.model_dir, current_time)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    
    # dataset = MusicDataset(args.dataset, SEGMENT_SIZE, 256)

    train_dir = os.path.join(args.dataset, 'train') 
    test_dir = os.path.join(args.dataset, 'test')

    train_dataset = MusicDataset(train_dir, SEGMENT_SIZE, 256, is_train=True)
    test_dataset = MusicDataset(test_dir, SEGMENT_SIZE, 256, is_train=False)

    print("train_dataset", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    model = UNet(in_channels=1, out_channels=4).to(DEVICE)
    
    if args.pretrained:
        model.load_state_dict(torch.load(args.pretrained))
    
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint))
    
    if args.loss == 'mse':
        loss_fn = nn.MSELoss()
    elif args.loss == 'l1':
        loss_fn = nn.L1Loss()
    
    spectral_loss_fn = SpectralLoss()

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2_reg)
    
    # train(args, model, dataloader, loss_fn, optimizer, writer)
    scheduler = None
    if args.use_cosine_annealing:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    train(args, model, train_loader, test_loader, loss_fn, spectral_loss_fn, optimizer, scheduler, writer)
    
    writer.close()
