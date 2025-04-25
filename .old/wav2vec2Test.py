import ai_edge_torch
import numpy as np
import torch
import torchaudio
import torchvision
import os

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

model_downloaded = os.path.exists("/home/dylenthomas/whisper/.model/Wav2Vec2.pt")
#device = "cuda" if torch.cuda.is_available() else "cpu"

if model_downloaded:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().eval()
else:
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = torch.jit.load(".model/Wav2Vec2.pt").eval()

#model = torchvision.models.resnet18(torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
#sample_inputs = (torch.randn(1, 3, 224, 224),)
#torch_outputs = model(*sample_inputs)

model = model.to(device="cpu")

#waveform, sample_rate = torchaudio.load("test.wav")
sample_inputs = (torch.rand(1, 16000),)
torch_outputs = model(*sample_inputs)

#if sample_rate != bundle.sample_rate:
#    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

#with torch.inference_mode():
#    torch_output, _ = model(waveform)

#decoder = GreedyCTCDecoder(labels=bundle.get_labels())
#transcript = decoder(torch_output[0])

#if model_downloaded:
#    print("saving the model...")
#    model_scripted = torch.jit.script(model)
#    model_scripted.save(".model/Wav2Vec2.pt")

# convert model to tflite
edge_model = ai_edge_torch.convert(model, sample_inputs)
edge_output = edge_model(*sample_inputs)

# validate the model
if np.allclose(torch_outputs.detach().numpy(), edge_output, atol=1e-5, rtol=1e-5):
    print("Inference result with PyTorch and TfLite was within tolerance")
else:
    print("Something wrong with PyTorch --> TfLite")

edge_model.export("/home/dylenthomas/whisper/.model/Wav2Vec2.tflite")