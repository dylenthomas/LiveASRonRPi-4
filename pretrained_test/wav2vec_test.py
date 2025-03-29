import torch
import torchaudio

import time
from torchaudio.utils import download_asset

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

device = torch.device("mps")
SPEECH_FILE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
print("Sample Rate: {}".format(bundle.sample_rate))
print("Labels: {}".format(bundle.get_labels()))

model = bundle.get_model().to(device)
print(model.__class__)

waveform, sample_rate = torchaudio.load(SPEECH_FILE)
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
    
with torch.inference_mode():
    emission, _ = model(waveform)

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])

model_scripted = torch.jit.script(model)
model_scripted.save("pretrained_asr_wav2vec2.pt")