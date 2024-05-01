import torch
import torch.nn as nn
import librosa

class SpectralLoss(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512, win_length=None):
        super(SpectralLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft

    def forward(self, y_pred, y_true):
    # y_pred 和 y_true shape (batch_size, num_sources, num_freq_bins, num_time_frames)
    
        # spec_pred = self._amp_to_db(y_pred)
        # spec_true = self._amp_to_db(y_true)

        spec_pred = torch.from_numpy(self._amp_to_db(y_pred))
        spec_true = torch.from_numpy(self._amp_to_db(y_true))

        # print("pred", spec_pred.shape)
        # print("true", spec_true.shape)
        # loss = nn.functional.mse_loss(spec_pred, spec_true)
        loss = torch.mean((spec_pred - spec_true) ** 2)

        return loss

    def _amp_to_db(self, spec):
        spec = spec.detach().cpu().numpy()
        return librosa.amplitude_to_db(spec, ref=1.0, amin=1e-7, top_db=80.0)    