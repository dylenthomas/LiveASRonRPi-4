import json
import librosa
import onnxruntime as ort
import torchaudio.transforms as T
import torch
import torch.nn as nn
import numpy as np

class LogMelSpectrogram():
    """Compute the log mel spectrogram of an audio waveform
    
        Audio must be sampled at 16,000 sr
        Audio must also be mono
    """
    def __init__(self, mel_filters:torch.Tensor, sample_rate:int, n_samples:int, n_fft:int, hop_length:int, device:str="cpu"):
        self.mel_floor = 1e-10

        self.mel_filters = mel_filters
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.device = device

        self.spectrogram_transform = T.Spectrogram(n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   power=2.0,
                                                   ).to(self.device)
        self.db_transform = T.AmplitudeToDB(stype="power").to(self.device)

    def convert_to_mel_scale(self, spectrogram):
        return torch.maximum(torch.Tensor([[self.mel_floor]]).to(device=self.device), torch.mm(self.mel_filters, spectrogram))

    def pad_or_trim_waveform(self, waveform):
        waveform_length = waveform.shape[-1]
        if waveform_length > self.n_samples:
            waveform = waveform[:-(waveform_length - self.n_samples)]
        elif waveform_length < self.n_samples:
            pad = (0, self.n_samples - waveform_length)
            waveform = nn.functional.pad(waveform, pad, "constant", value=0)
        return waveform
   
    def compute(self, x):
        #x = self.pad_or_trim_waveform(x)
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
    def __init__(self, config_path:str, special_tokens_path:str, vocab_path:str, device:str="cpu"):
        with open(config_path, 'r') as a:
            self.config = json.load(a)
            a.close()

        with open(special_tokens_path, 'r') as b:
            self.special_tokens = json.load(b)["added_tokens_decoder"]
            b.close()

        with open(vocab_path, 'r') as c:
            self.vocab = json.load(c)
            c.close()

        self.device = device

        self.sample_rate = self.config["sampling_rate"]
        self.hop_length = self.config["hop_length"]
        self.mel_filters = torch.Tensor(self.config["mel_filters"]).to(self.device)
        self.n_fft = self.config["n_fft"]
        self.n_samples = self.config["n_samples"]
        self.nb_max_frames = self.config["nb_max_frames"]

        self.lm_spect_transform = LogMelSpectrogram(self.mel_filters, self.sample_rate, self.n_samples, self.n_fft, self.hop_length, device=self.device)
    
    def decode_single(self, val):
        if val > 50257:
            return self.special_tokens[str(val)]["content"]
        else:
            return list(self.vocab.keys())[val] # val corresponds to the index

    def print_config(self):
        for key in self.config.keys():
            print(key)
            print("\t", self.config[key])

    def extract_features(self, waveform):
        features = self.lm_spect_transform.compute(waveform)
        if features.shape[-1] > self.nb_max_frames: # sometimes is 3001 and not 3000
            features = features[:, :-(features.shape[-1] - self.nb_max_frames)]
        return features
    
    def gen_decoder_ids(self):
        # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> 
        ids = [50258, 50259, 50359, 50363]
        return torch.tensor([ids], dtype=torch.long).to(self.device)

class onnxWraper():
    """
    Onnx wrapper for a VAD model, this model gives a percentage of certainty that there is human speech in each chunk of audio provided.
    The chunk length should be 512 for 16kHz audio, so with shape (10, 512) each of those 10 chunks will get a percentage chance that there is human speech in that audio.
    """
    def __init__(self, path, force_cpu=False):
        ort_opts = ort.SessionOptions()
        ort_opts.inter_op_num_threads = 1
        ort_opts.intra_op_num_threads = 1

        if "CUDAExecutionProvider" in ort.get_available_providers() and not force_cpu:
            self.inference_ses = ort.InferenceSession(path, providers=["CUDAExecutionProvider"], sess_options=ort_opts)
            print("onnx is using CUDA")
        else:
            self.inference_ses = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=ort_opts)
            print("onnx is using CPU")

        self._reset_states()

        self.num_samples = 512

        print("Onnx Inputs")
        for input_info in self.inference_ses.get_inputs():
            print("Input type name: {}".format(input_info.name))
            print("Input type shape: {}".format(input_info.shape))
            print("Input type info: {}".format(input_info.type))

        print("Onnx Outputs")
        for output_info in self.inference_ses.get_outputs():
            print("Output type name: {}".format(output_info.name))
            print("Output type shape: {}".format(output_info.shape))
            print("Output type info: {}".format(output_info.type))

    def _validate_shape(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() >  2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        return x

    def _reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_batch_size = 0

    def reset(self):
        self._reset_states(self._last_batch_size)

    def __call__(self, x):
        original_audio = x.clone()
        x = self._validate_shape(x)
        if x.shape[-1] != self.num_samples and x.shape[-1] % self.num_samples != 0:
            # reshape the data so the VAD model can process it
            remainder = x.shape[-1] % self.num_samples
            
            remainder_data = x[:, -remainder:] # set aside the remainder data
            x = x[:, :-remainder] # cut off the remainder data

            x = torch.reshape(x, (x.shape[-1] // self.num_samples, self.num_samples)) # reshape data
            remainder_data = nn.functional.pad(remainder_data, (0, self.num_samples-remainder), "constant", 0) # zero pad

            x = torch.cat((x, remainder_data))
        
        else:
            x = torch.reshape(x, (x.shape[-1] // self.num_samples, self.num_samples))
            
        batch_size = x.shape[0]
        context_size = 64

        if not self._last_batch_size:
            self._reset_states(batch_size)
        elif self._last_batch_size != batch_size:
            self._reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)

        ort_inputs = {"input": x.numpy(), "state": self._state.numpy(), "sr": np.array(16000, dtype="int64")}
        ort_outs = self.inference_ses.run(None, ort_inputs)
        out, state = ort_outs
        self._state = torch.from_numpy(state)

        self._context = original_audio[..., -context_size:]
        self._last_batch_size = batch_size

        return torch.from_numpy(out)