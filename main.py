from ctypes import * 

#import tensorflow as tf
import numpy as np
#from ai_edge_litert.interpreter import Interpreter
from transformers import WhisperForConditionalGeneration
import os
from utils.WhisperProcessor import offlineWhisperProcessor
import torchaudio
from scipy.io.wavfile import write

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
ptr = clib.accessMicrophone(b"default", 16000, 1, 512, 5, byref(sample_count))
mic_samples = [ptr[i] for i in range(sample_count.value)]
clib.freeBuffer(ptr)

signal = np.array(mic_samples, dtype=np.float32)
write("test_out.wav", 16000, signal)

#signal, _ = torchaudio.load("./8455-210777-0068.wav")

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