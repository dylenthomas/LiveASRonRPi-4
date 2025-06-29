import numpy as np
import os
import wave
import torch
import threading
import gensim
import nltk
import re 

from utils.faster_whisper import WhisperModel
from utils.LARS_utils import offlineWhisperProcessor, onnxWraper
from utils.kwHelper import kwVectorHelper
from ctypes import * 
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

processor = offlineWhisperProcessor(config_path="utils/configs/preprocessor_config.json",
                                    special_tokens_path="utils/configs/tokenizer_config.json",
                                    vocab_path="utils/configs/vocab.json", device=device
                                    )
kw_helper = kwVectorHelper()
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
model = WhisperModel("small", device=device, compute_type="float32")

#define c++ functions
clib_mic = CDLL("utils/micModule.so")
clib_serial = CDLL("utils/serialModule.so")

clib_mic.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int), c_float]
clib_mic.accessMicrophone.restype = POINTER(c_short)
clib_mic.freeBuffer.argtypes = [POINTER(c_short)]
clib_mic.freeBuffer.restype = None

clib_serial.openSerialPort.argtypes = [c_char_p]
clib_serial.openSerialPort.restype = c_int
clib_serial.configureSerialPort.argtypes = [c_int, c_int, c_int]
clib_serial.configureSerialPort.restype = c_bool
clib_serial.writeSerial.argtypes = [c_int, c_char_p, c_size_t]
clib_serial.writeSerial.restypes = c_int
clib_serial.closeSerial.argtypes = [c_int]
clib_serial.closeSerial.restypes = None

"""
This script runs three threads:
    thread 1 (main): analyzing audio for voice activation and running predcition 
    thread 2 (audio): collecting and storing audio buffers
    thread 3 (pi communication): parsing transcripts and sening commands to end devices
"""

def audio_loop():
    """
    audio collection loop to collect and store 2 audio buffers
        collected_seconds - the number of seconds of audio to collect from the microphone
    """
    print("Starting audio collection...")
    
    while running:
        ptr = clib_mic.accessMicrophone(mic_name, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count), a) # collect mic data
        buffer = np.ctypeslib.as_array(ptr, shape=(sample_count.value,)) # if issues add .copy()
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

def parse_prediction(transcription):
    """
    parse the transcript and use matrix/vector math to find words/pharases which are similar to the desired command keywords
    """ 
    transcription = " ".join(transcription)
    transcription = word_tokenize(transcription)

    # make a list of lists to get multiple word chunks that line up with the comand keyword vectors
    transcrpt_vecs = kw_helper.transcript2mat(transcription)

    # calculate the cosine similarity between the transcription and each keyword 
    # the matrix created will give the cosine similarity between each transcription and keyword, where
    #   a row is the similarity between a given chunk of the transcript and each keyword 
    found_keywords = []
    for i, kw_matrix in enumerate(vec_keyword_db):
        t_vec = transcrpt_vecs[i]

        # if there is a transcript shorter than some keywords there wont be a vector for it, so skip that length
        if len(t_vec) == 0: continue
        # if we get to one word keywords, but the transcript is longer than one word assume there is no intended keyword present
        if i == 2 and len(transcription) > 1: continue
        
        dot_prod = np.matmul(t_vec, kw_matrix) # get the dot product for each row and col

        t_norms = np.linalg.norm(t_vec, axis=1, keepdims=True)
        kw_norms = np.linalg.norm(kw_matrix, axis=0, keepdims=True)

        dot_prod = dot_prod / (t_norms * kw_norms)
        most_similar = np.argmax(dot_prod, axis=1) # find the index of the highest cosine similarity

        rows = np.arange(dot_prod.shape[0]) # make a vector to index each row
        passing_scores = dot_prod[rows, most_similar] > similarity_threshold
        if not np.any(passing_scores): continue # no found keywords 
        
        passing_inds = most_similar[passing_scores].tolist()
        for ind in passing_inds:
            found_kw = " ".join(transcription[ind:ind + (i + 1)]).lower()
            found_keywords.append(nl_keyword_db[found_kw])

    send_commands(found_keywords)

def send_commands(found_keywords):
    """
    Create a command packet with the keywords found in the transcription 
    The command packet is a bitfield as shown in supplementary/bitfield.txt

    https://www.rapidtables.com/convert/number/hex-to-binary.html?x=03
    """
    bitfield = 0x0000
    num_bytes = 2

    for ind in found_keywords:
        bitfield |= 0x0001 << ind
    
    if bitfield == 0: return # no keywords

    bitfield = bitfield.to_bytes(num_bytes, "big") # b'\x12\x34' ---> 0x1234
    #sent_bytes = clib_serial.writeSerial(serial, bitfield, num_bytes)
    #if sent_bytes == -1:
    #    print("There was error writing to serial.")
    #else:
    #    print("{} bytes where sent through serial".format(sent_bytes))

### SET VARIABLES ###
running = True

sample_count = c_int()
mic_name = b"plughw:CARD=Snowball"
sample_rate = 16000
channels = 1
buffer_frames = 1024
record_seconds = 1.0
a = 0.25 # the coefficient for running average
similarity_threshold = 0.8 # the threshold for the similarity between the transcription and the command

buffer = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)
buffer_que = [] # create a list to store if there is a buffer that needs to be analyzed

prediction_que = [] # store buffers to be predicted on in a list (queue)
clear_que = False
thres = 0.7 # voice activity threshold
energy_threshold = 0.001 # energy threshold to trigger prediction
rel_thres = 0.15 # percent of chunks with VA to consider speech
i = 0

nl_keyword_db = kw_helper.get_encodings()
vec_keyword_db = kw_helper.get_kw_mat()

serial_portname = b"/dev/tty"
serial_speed = 115200
expected_serial_bytes = 2
######

if __name__ == "__main__":
    #global serial # so send_commands() can access the serial file id

    #serial = clib_serial.openSerialPort(serial_portname)
    #if serial == -1:
    #    exit() # there was an issue opening the serial port
    #if not clib_serial.configureSerialPort(serial, serial_speed, expected_serial_bytes):
    #    exit() # there was an issue configuring the serial port
    
    executor = ThreadPoolExecutor(max_workers=int(30/record_seconds)) # the most threads that would be needed
    
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
                #save_audio(prediction_que)

                if clear_que:
                    prediction_que[:] = [] # clear the prediction que
                    i = 0
                    clear_que = False

                # send the prediction off to be parsed for keywords in another thread
                #executor.submit(parse_prediction, transcription)
                t = threading.Thread(target=parse_prediction, args=(transcription,), daemon=True) # args expects an iterable
                t.start()

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        executor.shutdown(wait=True)
        audio_thread1.join()
        #clib_serial.closeSerial(serial)