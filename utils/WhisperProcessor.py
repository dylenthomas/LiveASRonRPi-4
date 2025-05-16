import json
import torchaudio.transforms as T
import torch
import torch.nn as nn
import librosa

class LogMelSpectrogram():
    """Compute the log mel spectrogram of an audio waveform
    
        Audio must be sampled at 16,000 sr
        Audio must also be mono
    """
    def __init__(self, mel_filters:torch.Tensor, sample_rate:int, n_samples:int, n_fft:int, hop_length:int):
        self.mel_floor = 1e-10

        self.mel_filters = mel_filters
        self.sample_rate = sample_rate
        self.n_samples = n_samples

        self.spectrogram_transform = T.Spectrogram(n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   power=2.0,
                                                   )
        self.db_transform = T.AmplitudeToDB(stype="power")

    def convert_to_mel_scale(self, spectrogram):
        return torch.maximum(torch.Tensor([[self.mel_floor]]), torch.mm(self.mel_filters, spectrogram))

    def pad_or_trim_waveform(self, waveform):
        waveform_length = waveform.shape[-1]
        if waveform_length > self.n_samples:
            waveform = waveform[:-(waveform_length - self.n_samples)]
        elif waveform_length < self.n_samples:
            pad = (0, self.n_samples - waveform_length)
            waveform = nn.functional.pad(waveform, pad, "constant", value=0)
        return waveform
   
    def compute(self, x):
        x = self.pad_or_trim_waveform(x)
        x = self.spectrogram_transform(x)
        x = self.convert_to_mel_scale(x)

        # Got the rest from this function vvvvvv
        # /home/dylenthomas/miniconda3/envs/python310/lib/python3.10/site-packages/transformers/models/whisper/feature_extraction_whisper.WhisperFeatureExtractor._np_extract_fbank_features
        # not sure why they did it this way but this makes the spectrogram the same as their feature extractor

        x = torch.abs(x)
        x = torch.log10(x) # convert to log scale
        x = torch.maximum(x, torch.max(x) - 8.0)
        x = (x + 4.0) / 4.0
        
        return x

     
class offlineWhisperProcessor():
    """Offline processor for whisper decoding and feature extraction
    """
    def __init__(self, config_path:str, special_tokens_path:str, vocab_path:str):
        with open(config_path, 'r') as a:
            self.config = json.load(a)
            a.close()

        with open(special_tokens_path, 'r') as b:
            self.special_tokens = json.load(b)["added_tokens_decoder"]
            b.close()

        with open(vocab_path, 'r') as c:
            self.vocab = json.load(c)
            c.close()

        self.sample_rate = self.config["sampling_rate"]
        self.hop_length = self.config["hop_length"]
        self.mel_filters = torch.Tensor(self.config["mel_filters"])
        self.n_fft = self.config["n_fft"]
        self.n_samples = self.config["n_samples"]
        self.nb_max_frames = self.config["nb_max_frames"]

        self.lm_spect_transform = LogMelSpectrogram(self.mel_filters, self.sample_rate, self.n_samples, self.n_fft, self.hop_length)
    
    def decode_single(self, val):
        if val > 50257:
            return self.special_tokens[str(val)]["content"]
        else:
            return list(self.vocab.keys())[val] # val corresponds to the index

    def print_config(self):
        for key in self.config.keys():
            print(key)
            print("\t", self.config[key])

    def extract_features(self, waveform:torch.Tensor):
        features = self.lm_spect_transform.compute(waveform)
        if features.shape[-1] > self.nb_max_frames: # sometimes is 3001 and not 3000
            features = features[:, :-(features.shape[-1] - self.nb_max_frames)]
        return features.unsqueeze(0)
    

if __name__ == "__main__":
    processor = offlineWhisperProcessor("/home/dylenthomas/whisperProject/modelTest/whisper-tiny/preprocessor_config.json")

    waveform, _ = librosa.load("/home/dylenthomas/whisperProject/modelTest/audio/8455-210777-0068.wav", sr=16000, mono=True)
    waveform = torch.Tensor(waveform)

    waveform_features = processor.extract_features(waveform)
    print(waveform_features)