from ctypes import * 

#import tensorflow as tf
import numpy as np
#from ai_edge_litert.interpreter import Interpreter
from transformers import WhisperForConditionalGeneration
#import os
from utils.WhisperProcessor import offlineWhisperProcessor

#os.environ["TRANSFORMERS_OFFLINE"] = "1"

processor = offlineWhisperProcessor(config_path="utils/preprocessor_config.json",
                                    special_tokens_path="utils/tokenizer_config.json",
                                    vocab_path="utils/vocab.json"
                                    )

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
clib = CDLL("./micModule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int)]
clib.accessMicrophone.restype = POINTER(c_short)

clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

sample_count = c_int()
ptr = clib.accessMicrophone(b"microphone name", 16000, 1, 512, 5, byref(sample_count))
mic_samples = [ptr[i] for i in range(sample_count.value)]
clib.freeBuffer(ptr)

signal = np.array(mic_samples)
features = processor.extract_features(signal)
pred = model.generate(features)[0]

for tok in pred:
    print(processor.decode_single(tok))

#Wav2Vec2 = torch.jit.load("/home/dylenthomas/whisper/.model/Wav2Vec2.pt")
#decoder = GreedyCTCDecoder(labels)

#mic_samples = torch.tensor(mic_samples)
#with torch.inference_mode():
#    emission, _ = Wav2Vec2(mic_samples)

#transcript = decoder(emission[0])
#print(transcript)
