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
from utils.LARS_utils import kwVectorHelper
from ctypes import * 
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize

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

#define c++ functions
clib_mic = CDLL("utils/micModule.so")
#clib_serial = CDLL("utils/serialModule.so")

clib_mic.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int), c_float]
clib_mic.accessMicrophone.restype = POINTER(c_short)
clib_mic.freeBuffer.argtypes = [POINTER(c_short)]
clib_mic.freeBuffer.restype = None

#clib_serial.openSerialPort.argtypes = [c_char_p]
#clib_serial.openSerialPort.restype = c_int
#clib_serial.configureSerialPort.argtypes = [c_int, c_int, c_int]
#clib_serial.configureSerialPort.restype = c_bool
#clib_serial.writeSerial.argtypes = [c_int, c_char_p, c_size_t]
#clib_serial.writeSerial.restypes = c_int
#clib_serial.closeSerial.argtypes = [c_int]
#clib_serial.closeSerial.restypes = None

"""
This script runs three threads:
    thread 1 (main): analyzing audio for voice activation and running predcition 
    thread 2 (audio): collecting and storing audio buffers
    thread 3 (ui): a ui running on the server so I can see a live transcription and other information
    threads 4+ (pi communication): parsing transcripts and sening commands to end device
"""

def audio_loop():
    """
    audio collection loop to collect and store 2 audio buffers
        collected_seconds - the number of seconds of audio to collect from the microphone
    """
    print("Starting audio collection...")
    
    while running:
        ptr = clib_mic.accessMicrophone(mic_name, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count), a)
        buffer = np.ctypeslib.as_array(ptr, shape=(sample_count.value,))
        buffer = buffer.astype(np.float32) / 32768.0 # convert to float32
        clib_mic.freeBuffer(ptr)

        buffer_que.append(buffer)

def prediction(prediction_que, i = 0):
    transcription = []
    
    pred_array = np.concatenate(prediction_que)
    que_len = len(prediction_que)
    features = processor.extract_features(torch.from_numpy(pred_array).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")

    # Move cursor up by the number of segments (or clear screen if first run)
    if i == 0: # clear the screen if first run and move cursor to top left
        print("\033[2J \033[H")
    print('\033[F' * i, end='', flush=True)
    i = 0
    for segment in segments:
        print("[%.2fs -> %.2fs] \t%s" % (segment.start, segment.end, segment.text))
        cleaned_text = re.sub(r'[^\w\s]', '', segment.text) # remove all non characters and whitespace
        transcription.append(cleaned_text)
        i += 1

    return i, transcription

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
commands_sent = False

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
clear_que = False
thres = 0.7 # voice activity threshold
energy_threshold = 0.001 # energy threshold to trigger prediction
rel_thres = 0.15 # percent of chunks with VA to consider speech
i = 0

#serial_portname = b"/dev/tty"
#serial_speed = 115200
#expected_serial_bytes = 2
######

if __name__ == "__main__":
    #serial = clib_serial.openSerialPort(serial_portname)
    #if serial == -1:
    #    raise("There was an issue starting the serial port: {}".format(serial_portname))
    #if not clib_serial.configureSerialPort(serial, serial_speed, expected_serial_bytes):
    #    raise("There was an issue configuring the serial port")
    
    executor = ThreadPoolExecutor(max_workers=int(30/record_seconds)) # the most threads that would be needed
    kw_helper.setThreadManager(executor)

    audio_thread1 = threading.Thread(target=audio_loop, daemon=True)
    audio_thread1.start()

    try:
        while running:
            # check buffer for voice activity
            if len(buffer_que) > 0:
                vad_pred = vad_model(torch.from_numpy(buffer_que[0]))
                num_speech_chunks = torch.sum((vad_pred > thres).int()).item()
                rel_speech_chunks = num_speech_chunks / vad_pred.shape[0]
                audio_energy = torch.mean(torch.abs(torch.from_numpy(buffer_que[0])))

                if rel_speech_chunks > rel_thres and len(prediction_que) < int(30 / record_seconds) and audio_energy > energy_threshold:
                    prediction_que.append(buffer_que[0])

                elif len(prediction_que) > 0:
                    # no speech detected or there is no room left in the que
                    # if there is something in the que mark it for clearing and preform one last prediction
                    clear_que = True

                del buffer_que[0]

            if len(prediction_que) > 0:
                i, transcription = prediction(prediction_que, i)

                if clear_que:
                    prediction_que[:] = [] # clear the prediction que
                    i = 0
                    clear_que = False
                    # reset the variable stopping serial communication
                    commands_sent = False

                # send the prediction off to be parsed for keywords in another thread
                # run a thread to parse the prediciton each time the transcript is re made so that as soon as it detects a command after n runs it immediatley gets it
                # this should be fine assuming a low number of commands will be requested at a time and false positive rates are very low
                # then once a command packet is sent for this transcription chunk don't allow sending until the next chunk
                if not commands_sent:
                    kw_helper.addParsingThread(transcription)
                
                #t = threading.Thread(target=parse_prediction, args=(transcription,), daemon=True) # args expects an iterable
                #t.start()

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        executor.shutdown(wait=True)
        audio_thread1.join()
        #clib_serial.closeSerial(serial)