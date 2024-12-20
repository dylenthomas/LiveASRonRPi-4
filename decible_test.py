import numpy as np
from scipy.io.wavfile import read

A = 71.35
REF = 1e-4

def intensity(a):
    return (np.sqrt(np.mean(a**2)) / A)

a = read("./full_dataset/0/formual_1_9WKzo.wav")
a = np.array(a[1], dtype=float)

dB = 10*np.log10(intensity(a) / REF)
print(dB)

a = read("./full_dataset/3/target-whistle_xxvUR.wav")
a = np.array(a[1], dtype=float)

dB = 10*np.log10(intensity(a) / REF)
print(dB)