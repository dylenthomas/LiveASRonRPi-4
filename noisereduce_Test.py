from noisereduce.torchgate import TorchGate as TG
import wave
import pyaudio
from scipy.io import wavfile

p = pyaudio.PyAudio()

chunk = 1024

wf = wave.open("/Users/dylenthomas/Documents/whisper/full_dataset/0/backgound-noise_2IgBh.wav", 'rb')
rate, noisy_data = wavfile.read("/Users/dylenthomas/Documents/whisper/full_dataset/0/backgound-noise_2IgBh.wav")
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
                )

data = wf.readframes(chunk)
while data:
    stream.write(data)
    data = wf.readframes(chunk)
    
tg = TG(sr=wf.getframerate(), nonstationary=True).to('cpu')
#reduced_audio = tg(audio_tensor)


wf = wave.open("test.wav", 'rb')
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True
                )

data = wf.readframes(chunk)
while data:
    stream.write(data)
    data = wf.readframes(chunk)