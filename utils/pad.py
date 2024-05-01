import torch
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
        return torch.nn.functional.pad(signal, (left, right), mode='reflect'), (left, right)
    else:
        return signal, (0, 0)
