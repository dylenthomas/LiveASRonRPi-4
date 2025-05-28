from ctypes import * 
import ctypes
#import tensorflow as tf
import numpy as np
#from ai_edge_litert.interpreter import Interpreter
from transformers import WhisperForConditionalGeneration
import os
from utils.WhisperProcessor import offlineWhisperProcessor
import torchaudio
from scipy.io.wavfile import write
import wave
import torch

#os.environ["TRANSFORMERS_OFFLINE"] = "1"

processor = offlineWhisperProcessor(config_path="utils/preprocessor_config.json",
                                    special_tokens_path="utils/tokenizer_config.json",
                                    vocab_path="utils/vocab.json"
                                    )

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True)

clib = CDLL("./utils/micModule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int)]
clib.accessMicrophone.restype = POINTER(c_short)

clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

sample_count = c_int()
ptr = clib.accessMicrophone(b"plughw:CARD=Snowball", 16000, 1, 1024, 5, byref(sample_count))

signal = np.ctypeslib.as_array(ptr, shape=(sample_count.value,)).copy()
signal = signal.astype(np.float32) / 32768.0
signal = torch.from_numpy(signal)

clib.freeBuffer(ptr)

#wav = wave.open("test_out.wav", "wb")
#wav.setnchannels(1)
#wav.setsampwidth(2)  # 16-bit
#wav.setframerate(16000)
#wav.writeframes(signal)
#wav.close()

#signal, _ = torchaudio.load("./test_out.wav")
#signal = signal.resize(signal.shape[-1])

features = processor.extract_features(signal)
pred = model.generate(features, language="en")[0]

transcription = []
for i, tok in enumerate(pred):
    transcription.append(processor.decode_single(tok))

transcription = "".join(transcription).replace("Ġ", " ")
if transcription[0] == " ":
    transcription =  transcription[1:]

print("===Transcription===")
print(transcription)