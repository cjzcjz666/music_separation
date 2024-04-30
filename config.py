import torch

# WINDOW_SIZE = 2047
# HOP_LENGTH = 512
# SAMPLING_RATE = 22050

WINDOW_SIZE = 2047
HOP_LENGTH = 512  # WINDOW_SIZE // 4
SAMPLING_RATE = 22050
SEGMENT_SIZE = 512  # SAMPLING_RATE * 10

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")