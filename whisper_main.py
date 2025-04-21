import torch
import torchaudio
from ctypes import * 

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

labels = ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')

clib = CDLL("/home/dylenthomas/whisper/whispermodule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int)]
clib.accessMicrophone.restype = POINTER(c_short)

clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

sample_count = c_int()
ptr = clib.accessMicrophone(b"microphone name", 16000, 1, 512, 5, byref(sample_count))
mic_samples = [ptr[i] for i in range(sample_count.value)]
clib.freeBuffer(ptr)

Wav2Vec2 = torch.jit.load("/home/dylenthomas/whisper/.model/Wav2Vec2.pt")
decoder = GreedyCTCDecoder(labels)


mic_samples = torch.tensor(mic_samples)
with torch.inference_mode():
    emission, _ = Wav2Vec2(mic_samples)

transcript = decoder(emission[0])
print(transcript)