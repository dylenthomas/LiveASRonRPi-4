from noisereduce.torchgate import TorchGate as TG
import torchaudio
from pydub import AudioSegment, effects

def normalize(dir):
    rawsound = AudioSegment.from_file(dir, 'wav')
    normalized_sound = effects.normalize(rawsound)
    normalized_sound.export(dir, format='wav')

audio_tensor, sr = torchaudio.load("./UnusedData/full_dataset_copy/0/background-noise_b6P2L.wav")
tg = TG(sr=sr, nonstationary=False).to('cpu')
reduced_audio = tg(audio_tensor)
torchaudio.save("test.wav", reduced_audio, sr)

normalize("test.wav")