# import os
# import numpy as np
# import torch
# from torch.utils.data import Dataset
# from data.preprocess import preprocess_data

# class MusicDataset(Dataset):
#     def __init__(self, data_dir, segment_size):
#         self.data_dir = data_dir
#         self.segment_size = segment_size
        
#         if os.path.exists(data_dir) and os.path.isdir(data_dir):
#             wav_files = self._find_wav_files(data_dir)
#             if wav_files:
#                 print("Converting WAV files to NPZ...")
#                 preprocess_data(data_dir)
#                 self.data_dir = '../musdb18_npz/train'
#                 print("Finish preprocess")
#         self.file_list = os.listdir(self.data_dir)
    
#     def _find_wav_files(self, directory):
#         wav_files = []
#         for root, dirs, files in os.walk(directory):
#             for file in files:
#                 if file.endswith('.wav'):
#                     wav_files.append(os.path.join(root, file))
#         return wav_files

#     def __len__(self):
#         return len(self.file_list)
    
#     def __getitem__(self, index):
#         file_name = self.file_list[index]

#         # print("file_name", file_name)
#         file_path = os.path.join(self.data_dir, file_name)

#         # print("file_path", file_path)
        
#         data = np.load(file_path)
#         mix = data['mix']
#         target = np.stack([data['bass'], data['drums'], data['other'], data['vocals']], axis=0)

#         if mix.shape[1] < self.segment_size:
#             # padding
#             padding = self.segment_size - mix.shape[1]
#             mix = np.pad(mix, ((0, 0), (0, padding)), 'constant', constant_values=(0, 0))
#             target = np.pad(target, ((0, 0), (0, 0), (0, padding)), 'constant', constant_values=(0, 0))

#         # print("mix", mix.shape[1])
#         # print("size", self.segment_size)

#         if mix.shape[1]==self.segment_size:
#             start = 0
#         else:
#             start = np.random.randint(0, mix.shape[1] - self.segment_size)
#         end = start + self.segment_size
        
#         mix_segment = mix[:, start:end]
#         target_segment = target[:, :, start:end]
        
#         return torch.from_numpy(mix_segment), torch.from_numpy(target_segment)

import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from data.preprocess import preprocess_data

class MusicDataset(Dataset):
    def __init__(self, data_dir, segment_size, output_len, is_train=True):
        self.data_dir = data_dir
        self.segment_size = segment_size
        self.output_len = output_len
        self.is_train = is_train
        self.segments = []

        if os.path.exists(data_dir) and os.path.isdir(data_dir):
            wav_files = self._find_wav_files(data_dir)
            if wav_files:
                print("Converting WAV files to NPZ...")
                self.data_dir = preprocess_data(data_dir, self.is_train)
                print("Finish preprocessing")

            npz_files = [f for f in os.listdir(self.data_dir) if f.endswith('.npz')]
            for npz_file in npz_files:
                file_path = os.path.join(self.data_dir, npz_file)
                data = np.load(file_path)
                # print(type(data))
                mix = data['mix']
                target = np.stack([data['bass'], data['drums'], data['other'], data['vocals']], axis=0)

                if self.is_train:
                    # hop = self.segment_size // 4
                    # for start in range(0, mix.shape[1] - self.segment_size + 1, hop):
                    #     mix_segment = mix[:, start:start + self.segment_size]
                    #     target_segment = target[:, :, start:start + self.segment_size]
                    #     self.segments.append((mix_segment, target_segment))
                    
                    hop = self.segment_size // 4   # hop 128
                    for n in range((mix.shape[1] - self.segment_size) // hop + 1):
                        start = n * hop
                        mix_segment = mix[:, start:start + self.segment_size]  # [1024,512]
                        target_segment = target[:, :, start:start + self.segment_size] # [4,1024,512]
                        self.segments.append((mix_segment, target_segment))
                else:
                    self.segments.append((mix, target))
             # (7363, 2)

    def _find_wav_files(self, directory):
        wav_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith('.wav'):
                    wav_files.append(os.path.join(root, file))
        return wav_files

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, index):
        mix_segment, target_segment = self.segments[index]
        # import pdb;
        # pdb.set_trace()
        if self.is_train:
            length = mix_segment.shape[1]  # 512
            # if length > self.output_len:
            start = random.randint(0, length - self.output_len)
            mix_segment = mix_segment[:, start:start + self.output_len]
            target_segment = target_segment[:, :, start:start + self.output_len]

        return torch.from_numpy(mix_segment), torch.from_numpy(target_segment)