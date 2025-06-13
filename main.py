from ctypes import * 
import numpy as np
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
import os
from utils.LARS_utils import offlineWhisperProcessor, onnxWraper
import torchaudio
from scipy.io.wavfile import write
import wave
import torch
import threading
import timeit
import inspect
from faster_whisper import WhisperModel
import sys

os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
processor = offlineWhisperProcessor(config_path="utils/preprocessor_config.json",
                                    special_tokens_path="utils/tokenizer_config.json",
                                    vocab_path="utils/vocab.json",
                                    device=device
                                    )
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True, device_map="auto")
model = WhisperModel("small", device=device, compute_type="float32")
clib = CDLL("./utils/micModule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int), c_float]
clib.accessMicrophone.restype = POINTER(c_short)
clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

"""
Run three threads:
    thread 1 (main): analyzing audio for voice activation and running predcition 
    thread 2 (audio): collecting and storing audio buffers
    thread 3 (pi communication): sending commands and data to end deevices
"""

def audio_loop():
    """
    audio collection loop to collect and store 2 audio buffers
        collected_seconds - the number of seconds of audio to collect from the microphone
    """
    print("Starting audio collection...")
    
    while running:
        ptr = clib.accessMicrophone(mic_name, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count), a) # collect mic data
        buffer = np.ctypeslib.as_array(ptr, shape=(sample_count.value,)) # if issues add .copy()
        buffer = buffer.astype(np.float32) / 32768.0 # convert to float32
        clib.freeBuffer(ptr)

        buffer_que.append(buffer)

def prediction(prediction_que):
    pred_array = np.concatenate(prediction_que)
    que_len = len(prediction_que)
    features = processor.extract_features(torch.from_numpy(pred_array).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")
    
    for segment in segments:
        print("[Window Que Length = %d] ===Transcription=== : [%.2fs -> %.2fs] \t%s" % (que_len, segment.start, segment.end, segment.text))

def save_audio(audio):
    window = np.concatenate(audio)
    window = window * (2**15)
    window = window.astype(dtype=np.int16)
    wav = wave.open("test_out.wav", "wb")
    wav.setnchannels(1)
    wav.setsampwidth(2)  # 16-bit
    wav.setframerate(16000)
    wav.writeframes(window)
    wav.close()
        
### SET VARIABLES ###
running = True

sample_count = c_int()
mic_name = b"plughw:CARD=Snowball"
sample_rate = 16000
channels = 1
buffer_frames = 1024
record_seconds = 1.0
a = 0.25 # the coefficient for running average

buffer = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)
buffer_que = [] # create a list to store if there is a buffer that needs to be analyzed

prediction_que = [] # store buffers to be predicted on in a list (queue)
que_ready = False
thres = 0.7 # voice activity threshold

attention_mask = torch.ones(1, 80)
######

audio_thread = threading.Thread(target=audio_loop, daemon=True)
audio_thread.start()

try:
    while running:
        # check buffer for voice activity
        if len(buffer_que) > 0:
            vad_pred = vad_model(torch.from_numpy(buffer_que[0]))
            num_speech_chunks = torch.sum((vad_pred > thres).int()).item()

            if num_speech_chunks > 0 and len(prediction_que) < int(30 / record_seconds):
                # if speech is detected and there is room in the que
                prediction_que.append(buffer_que[0])
                print(len(prediction_que))
            elif len(prediction_que) > 0:
                # no speech detected or there is no room left in the que
                # it there is something in the que mark it ready for prediction
                que_ready = True

            del buffer_que[0]

        if que_ready:
            prediction(prediction_que)
            #save_audio(prediction_que)

            prediction_que = []
            que_ready = False            

except KeyboardInterrupt:
    print("\nStopping...")
    running = False
    audio_thread.join()