import torch
import torchaudio
import random
import os
import numpy as np
from pydub import AudioSegment, effects

SR = 44100
NUM_SAMPLES = SR
OPTIONS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz12345678910'
A = 71.35
REF = 1e-4

def intensity(a):
    return (np.sqrt(np.mean(a**2)) / A)

def load_wav_file(file_name, dir):
    audio_tensor, sample_rate = torchaudio.load(dir)
    
    if sample_rate != SR:
        resampler = torchaudio.transforms.Resample(sample_rate, SR)
        audio_tensor = resampler(audio_tensor)
         
    if audio_tensor.shape[1] > NUM_SAMPLES:
        ref_point = audio_tensor.shape[1] // 2
        audio_tensor = audio_tensor[:, ref_point-SR:ref_point]
    elif audio_tensor.shape[1] < NUM_SAMPLES:
        missing = NUM_SAMPLES - audio_tensor.shape[1]
        audio_tensor = torch.nn.functional.pad(audio_tensor, (0, missing))
    
    base = '' 
    for _ in range(5):
        base += OPTIONS[random.randrange(0, len(OPTIONS) - 1)]
    
    torchaudio.save(os.path.join("/Users/dylenthomas/Documents/whisper/full_dataset/2", base + file_name), audio_tensor, SR)\

def normalize(dir):
    rawsound = AudioSegment.from_file(dir, 'wav')
    normalized_sound = effects.normalize(rawsound)
    normalized_sound.export(dir, format='wav')
        
if __name__ == '__main__':
    files = {}
    for (dirpath, dirnames, filenames) in os.walk("/Users/dylenthomas/Documents/whisper/full_dataset"):
        for filename in filenames:
            if filename.endswith('.wav'):
                #work with music file names
                if len(filename.split('.')) > 2:
                    filesplit = filename.split('.')
                    a = ''
                    for i in range(len(filesplit) - 1):
                        a += filesplit[i]
                    new_filename = a + '.wav'

                    os.rename(os.sep.join([dirpath, filename]), os.sep.join([dirpath, new_filename]))
                    filename = new_filename
                    
                files[filename] = os.sep.join([dirpath, filename])
    
    for file in files:
        dir = files[file]
        normalize(dir)