from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
import os
import glob

class dataset(Dataset):
    def __init__(self, root_path: str, sample_rate: int, num_samples: int, device='cpu'):
        self.class_paths = glob.glob(os.path.join(root_path, '*'))
        self.data = []

        for path in self.class_paths:
            paths = glob.glob(os.path.join(path, '*.wav'))
            for input_path in paths:
                self.data.append(input_path)
            
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.device = device
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=978, #computed to make the size of data 91
            n_mels=64
        ).to(self.device)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, indx):
        sample_path = self.data[indx]
        
        # get class int
        sample_path_split = sample_path.split('/')
        for section in sample_path_split:
            try:
                sample_class = int(section)
            except ValueError:
                pass
                
        
        """data_seen = self.data_per_class[0]
        for i, data in enumerate(self.data):
            if indx < data_seen and i == 0:
                sample_path = data[indx]
                sample_class = i
                break
            elif indx < data_seen and i > 0:
                sample_path = data[indx - self.data_per_class[i - 1]]
                sample_class = i
                break
            else:
                data_seen += self.data_per_class[i + 1]
"""
        signal, sample_rate = torchaudio.load(sample_path)
        signal = signal.to(self.device)

        if sample_rate != self.sample_rate:
            signal = self._resample(signal, sample_rate)

        signal = self._edit_signal_length(signal)
        signal = self.mel_transform(signal)
        signal = signal.view(64, 91)

        return signal, sample_class


    def _resample(self, signal, sample_rate):
        resampler = torchaudio.transforms.Resample(sample_rate, self.sample_rate).to(self.device)
        return resampler(signal)

    def _edit_signal_length(self, signal):
        signal_len = signal.shape[1]
        if signal_len > self.num_samples:
            return signal[:, :self.num_samples]
        elif signal_len < self.num_samples:
            missing_samples = self.num_samples - signal_len
            last_dim_padding = (0, missing_samples)
            return torch.nn.functional.pad(signal, last_dim_padding)


if __name__ == '__main__':
    root = './full_dataset'
    sample_rate = 44100
    num_samples = 44100

    rtrdDS = dataset(root, sample_rate, num_samples, device='cpu')
    length = rtrdDS.__len__()
    item, class_ = rtrdDS.__getitem__(2)

    print(length)
    print(item.shape, class_)
