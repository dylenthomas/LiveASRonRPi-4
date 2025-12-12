import numpy as np
import os
import wave
import torch
import threading
import re
import datetime

from utils.faster_whisper import WhisperModel
from utils.ML_utils import offlineWhisperProcessor, onnxWraper
from utils.LARS_utils import TCPCommunication
from ctypes import * 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from sentence_transformers import SentenceTransformer, util

### setup relevant helper classes ------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

print("Starting whisper processor...")
processor = offlineWhisperProcessor(config_path="utils/configs/preprocessor_config.json",
                                    special_tokens_path="utils/configs/tokenizer_config.json",
                                    vocab_path="utils/configs/vocab.json", device=device
                                    )
print("Starting sentence transformer...")
word_vector_generator = SentenceTransformer("all-MiniLM-L6-v2")
word_vector_generator.to(device)
print("Starting VAD model...")
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
print("Starting Whisper...")
model = WhisperModel("small", device=device, compute_type="float32")
tcpCommunicator = TCPCommunication()
### -------------------------------------------------------------------------------------------------

### define c++ functions ----------------------------------------------------------------------------

# MAKE A SO LIBRARY FOR EACH MIC
clib_mic = CDLL("utils/micModule.so")

clib_mic.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_float, POINTER(c_int)]
clib_mic.accessMicrophone.restype = POINTER(c_float)
clib_mic.freeBuffer.argtypes = [POINTER(c_float)]
clib_mic.freeBuffer.restype = None
clib_mic.close_mic.argtypes = [c_char_p]
clib_mic.close_mic.restype = None
### -------------------------------------------------------------------------------------------------

def audio_loop1():
    print("Starting mic1...")
    buffer1 = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)

    while running:
        ptr1 = clib_mic.accessMicrophone(mic_name1, sample_rate, channels, buffer_frames, record_seconds, byref(sample_count1))
       
        buffer1 = np.ctypeslib.as_array(ptr1, shape=(sample_count1.value,))
        clib_mic.freeBuffer(ptr1)

        mics[0].append(buffer1)

    clib_mic.close_mic(mic_name1)

def audio_loop2():
    print("Starting mic2...")
    buffer2 = np.zeros(int(record_seconds * sample_rate), dtype=np.float32)

    while running:
        ptr2 = clib_mic.accessMicrophone(mic_name2, sample_rate, channels, buffer_frames, intrecord_seconds, byref(sample_count2))

        buffer2 = np.ctypeslib.as_array(ptr2, shape=(sample_count2.value,))
        clib_mic.freeBuffer(ptr2)
               
        mics[1].append(buffer2)

    clib_mic.close_mic(mic_name2)

def transcribe(prediction_que):
    transcription = []
    
    pred_array = np.concatenate(prediction_que)
    que_len = len(prediction_que)
    features = processor.extract_features(torch.from_numpy(pred_array).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")

    for segment in segments:
        cleaned_text = segment.text.lstrip()
        print("[%.2fs -> %.2fs]-%s" % (segment.start, segment.end, cleaned_text))
        transcription.append(cleaned_text)

    return transcription

def parse_transcript(transcript):
    packet = ''

    transcript = [" ".join(transcript)]
    transcript_embedding = word_vector_generator.encode(transcript)
    transcript_embedding = np.transpose(transcript_embedding)

    similarities = np.matmul(keyword_embeddings, transcript_embedding)
    max_similarity_ind = np.argmax(similarities).item()
    max_similarity = similarities[max_similarity_ind]

    if max_similarity > vector_similarity_thres:
        keyword = flattened_keywords[max_similarity_ind]
        packet += f'{keyword},'

    return packet

def save_audio(audio, name):
    window = audio * (2**15)
    window = window.astype(dtype=np.int16)
    wav = wave.open("{}.wav".format(name), "wb")
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
mic_name2 = b"plughw:CARD=Snowball_1"
record_seconds = 0.5

prediction_que = []
longestWhisperBuffer = 30
prediction_que_max_length = int(longestWhisperBuffer // record_seconds)

mics = [[], []]

VAD_threshold = 0.7
VAD_percent_speech_threshold = 0.1
VAD_reset_time = 1.0

ONE = ["lights", "mute", "unmute"]
TWO = ["lights on", "lights off", "volume down", "volume up"]
THREE = ["overhead lamp off","overhead lamp on", "desk lights off", "desk lights on", "set aux audio", "set phono audio"]
keywords = [THREE, TWO, ONE]
keyword_embeddings = np.concatenate([word_vector_generator.encode(keyword_group) for keyword_group in keywords])
flattened_keywords = [x for sub in keywords for x in sub]
vector_similarity_thres = 0.8
### ----------------------------------------------------------------------------------------

if __name__ == "__main__":    
    print("Waiting to connect to end device...")
    #tcpCommunicator.connectClient()
    print("Connected to end device.")

    executor = ThreadPoolExecutor(max_workers=prediction_que_max_length)

    audio_thread1 = threading.Thread(target=audio_loop1)
    audio_thread1.start()

    audio_thread2 = threading.Thread(target=audio_loop2)
    audio_thread2.start()

    start = datetime.datetime.now()

    clear_prediction_que = False
    already_sent = []

    try:
        while running:
            if len(mics[0]) == 0 or len(mics[1]) == 0:
                continue

            mic1_buffer = mics[0][0]
            mic2_buffer = mics[1][0]

            correlation = np.correlate(mic1_buffer, mic2_buffer, 'full')
            delay = np.argmax(correlation) - (len(mic2_buffer) - 1)
            if delay > 0:
                mic2_buffer = np.concatenate([np.zeros(delay), mic2_buffer])
            else:
                mic2_buffer = mic2_buffer[-delay:]

            combined_buffer = 0.5 * (mic1_buffer + mic2_buffer)
            
            VAD_pred = vad_model(torch.from_numpy(combined_buffer))
            print(torch.mean(VAD_pred))
            number_speech_chunks = torch.sum((VAD_pred > VAD_threshold).int()).item()
            percent_speech_chunks = number_speech_chunks / VAD_pred.shape[0]
            print(percent_speech_chunks)
            speech_in_audio = percent_speech_chunks >= VAD_percent_speech_threshold

            if speech_in_audio:
                prediction_que.append(combined_buffer)
            elif len(prediction_que) > 0:
                clear_prediction_que = True
            
            if len(prediction_que) > prediction_que_max_length:
                prediction_que = prediction_que[-prediction_que_max_length:]
                clear_prediction_que = True

            if len(prediction_que) > 0:
                transcript = transcribe(prediction_que)
                packet = parse_transcript(transcript)
                packet = packet.split(",")

                packet_to_send = []
                
                for command in packet:
                    if command not in already_sent:
                        packet_to_send.append(command)
                        already_sent.append(command)

                packet_to_send = ",".join(packet_to_send)

                if packet_to_send != '':
                    executor.submit(tcpCommunicator.sendToServer, packet_to_send)

            if clear_prediction_que:
                prediction_que[:] = []
                already_sent[:] = []
                clear_prediction_que = False

            del mics[0][0]
            del mics[1][0]

            if (datetime.datetime.now() - start).seconds >= VAD_reset_time:
                start = datetime.datetime.now()
                vad_model.reset()
                
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        #executor.shutdown(cancel_futures=True)
        audio_thread1.join()
        audio_thread2.join()
        tcpCommunicator.closeClientConnection()