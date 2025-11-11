import numpy as np
import os
import wave
import torch
import threading
import gensim
import nltk
import re 

from utils.faster_whisper import WhisperModel
from utils.ML_utils import offlineWhisperProcessor, onnxWraper
from utils.LARS_utils import kwVectorHelper, TCPCommunication
from ctypes import * 
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize

### setup relevant helper classes ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

print("Starting whisper processor...")
processor = offlineWhisperProcessor(config_path="utils/configs/preprocessor_config.json",
                                    special_tokens_path="utils/configs/tokenizer_config.json",
                                    vocab_path="utils/configs/vocab.json", device=device
                                    )
print("Creating Vectors for commands..")
kw_helper = kwVectorHelper()
print("Starting VAD model...")
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
print("Starting Whisper...")
model = WhisperModel("small", device=device, compute_type="float32")
tcpCommunicator = TCPCommunication()
### -------------------------------------------------------------------------------------------------

### define c++ functions ----------------------------------------------------------------------------
clib_mic = CDLL("utils/micModule.so")

clib_mic.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int), c_float]
clib_mic.accessMicrophone.restype = POINTER(c_short)
clib_mic.freeBuffer.argtypes = [POINTER(c_short)]
clib_mic.freeBuffer.restype = None
### -------------------------------------------------------------------------------------------------

"""
This script runs three threads:
    thread 1 (main): analyzing audio for voice activation and running predcition 
    thread 2 (audio): collecting and storing audio buffers
    thread 3 (ui): a ui running on the server so I can see a live transcription and other information
    threads 4+ (pi communication): parsing transcripts and sening commands to end device
"""

def audio_loop():
    print("Starting audio collection...")
    
    while running:
        ptr1 = clib_mic.accessMicrophone(mic_name1, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count1), runningAvgCoef)
        ptr2 = clib_mic.accessMicrophone(mic_name2, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count2), runningAvgCoef)

        buffer1 = np.ctypeslib.as_array(ptr1, shape=(sample_count1.value,))
        buffer1 = buffer1.astype(np.float32) / 32768.0 # convert to float32
        clib_mic.freeBuffer(ptr1)

        buffer2 = np.ctypeslib.as_array(ptr2, shape=(sample_count2.value,))
        buffer2 = buffer2.astype(np.float32) / 327680.0 #convert to float32
        clib_mic.freeBuffer(ptr2)

        buffer_que[0].append(buffer1)
        buffer_que[1].append(buffer2)

def prediction(prediction_que, i = 0):
    transcription = []
    
    pred_array = np.concatenate(prediction_que)
    que_len = len(prediction_que)
    features = processor.extract_features(torch.from_numpy(pred_array).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")

    for segment in segments:
        print("[%.2fs -> %.2fs] \t%s" % (segment.start, segment.end, segment.text))
        cleaned_text = re.sub(r'[^\w\s]', '', segment.text) # remove all non characters and whitespace
        transcription.append(cleaned_text)

    return transcription

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

### SET VARIABLES -------------------------------------------------------------------------
running = True

sample_count1 = c_int()
sample_count2 = c_int()
mic_name1 = b"plughw:CARD=Snowball"
mic_name2 = b"plughw:CARD="
sample_rate = 16000
channels = 1
buffer_frames = 1024
record_seconds = 1.0
runningAvgCoef = 0.25

longestWhisperBuffer = 30           # max seconds of audio whisper can predict on

buffer1 = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)
buffer2 = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)
buffer_que = [[], []]

prediction_que = []                 # store buffers to be predicted on in a list (queue)
clear_que = False
thres = 0.7                         # voice activity threshold
energy_threshold = 0.001            # energy threshold to trigger prediction
rel_thres = 0.15                    # percent of chunks with VA to consider speech
### ----------------------------------------------------------------------------------------

if __name__ == "__main__":    
    print("Waiting to connect to end device...")
    tcpCommunicator.connectClient()
    print("Connected to end device.")

    executor = ThreadPoolExecutor(max_workers=int(longestWhisperBuffer/record_seconds)) # the most threads that would be needed

    audio_thread1 = threading.Thread(target=audio_loop, daemon=True)
    audio_thread1.start()

    try:
        while running:
            # check buffer for voice activity
            # predict on the buffer with the highest voice activity

            most_speech_chunks = 0
            most_audio_energy = 0
            best_mic_index = None

            for mic in len(buffer_que):
                buffer = buffer_que[mic][0]
                vad_pred = vad_model(torch.from_numpy(buffer))
                num_speech_chunks = torch.sum((vad_pred > thres).int()).item()

                rel_speech_chunks = num_speech_chunks / vad_pred.shape[0]
                audio_energy = torch.mean(torch.abs(torch.from_numpy(buffer)))

                if rel_speech_chunks > most_speech_chunks and audio_energy > most_audio_energy:
                    most_speech_chunks = rel_speech_chunks
                    most_audio_energy = audio_energy
                    best_mic_index = mic

            if most_speech_chunks > rel_thres and len(prediction_que) < int(longestWhisperBuffer / record_seconds) and most_audio_energy > energy_threshold:
                prediction_que.append(buffer_que[best_mic_index][0])
            elif len(prediction_que) > 0:
                # no speech detected or there is no room left in the que
                # if there is something in the que mark it for clearing and preform one last prediction
                clear_que = True

            # clear the buffers
            for mic in len(buffer_que):
                del buffer_que[mic][0]

            if len(prediction_que) > 0:
                transcription = prediction(prediction_que)

                if clear_que:
                    # clear the prediction que
                    prediction_que[:] = []
                    i = 0
                    clear_que = False
                    # reset the variable stopping tcp communication
                    tcpCommunicator.command_sent = False

                # send the prediction off to be parsed for keywords in another thread
                # run a thread to parse the prediciton each time the transcript is re made so that as soon as a command is detected after n runs it will immediately be sent
                if not tcpCommunicator.command_sent:
                    DEBUG = False # if you want to see errors in the threads set to true so the regular threading class is used. 
                    if DEBUG:
                        t = threading.Thread(target=kw_helper.parse_prediction, args=(transcription, tcpCommunicator), daemon=True) # args expects an iterable
                        t.start()
                    else:
                        executor.submit(kw_helper.parse_prediction, (transcription, tcpCommunicator))
                
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        executor.shutdown(wait=True)
        audio_thread1.join()
        tcpCommunicator.closeClientConnection()