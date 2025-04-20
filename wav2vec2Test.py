import torch
import torchaudio
import matplotlib.pyplot as plt

class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        indices = torch.argmax(emission, dim=-1) #[num, seq]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

device = "cuda" if torch.cuda.is_available() else "cpu"

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

waveform, sample_rate = torchaudio.load("test.wav")
waveform = waveform.to(device)

if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

with torch.inference_mode():
    emission, _ = model(waveform)

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)

print("saving the model...")
model_scripted = torch.jit.script(model)
model_scripted.save(".model/Wav2Vec2.pt")