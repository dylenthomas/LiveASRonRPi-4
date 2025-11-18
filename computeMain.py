import numpy as np
import os
import wave
import torch
import threading
import re 

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
clib_mic = CDLL("utils/micModule.so")

clib_mic.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int), c_float]
clib_mic.accessMicrophone.restype = POINTER(c_short)
clib_mic.freeBuffer.argtypes = [POINTER(c_short)]
clib_mic.freeBuffer.restype = None
### -------------------------------------------------------------------------------------------------

def audio_loop():
    print("Starting audio collection...")
    
    while running:
        ptr1 = clib_mic.accessMicrophone(mic_name1, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count1), runningAvgCoef)
        #ptr2 = clib_mic.accessMicrophone(mic_name2, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count2), runningAvgCoef)

        buffer1 = np.ctypeslib.as_array(ptr1, shape=(sample_count1.value,))
        buffer1 = buffer1.astype(np.float32) / 32768.0 # convert to float32
        clib_mic.freeBuffer(ptr1)

        #buffer2 = np.ctypeslib.as_array(ptr2, shape=(sample_count2.value,))
        #buffer2 = buffer2.astype(np.float32) / 327680.0 #convert to float32
        #clib_mic.freeBuffer(ptr2)

        mics[0].append(buffer1)
        mics[1].append(buffer1)
        #mics[1].append(buffer2)

def prediction(prediction_que):
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
    transcript = [" ".join(transcript)]
    transcript_embedding = word_vector_generator.encode(transcript)

    max_similarity = 0.0
    max_similarity_ind = (0, 0)

    for i in range(len(keyword_embeddings)):
        kw_matrix = keyword_embeddings[i]

        similarities = util.cos_sim(transcript_embedding, kw_matrix)
        most_similar = torch.argmax(similarities)
            
        if similarities[0, most_similar] > max_similarity:
            max_similarity = similarities[0, most_similar]
            max_similarity_ind = (i, most_similar.item())

    if max_similarity > vector_similarity_thres:
        most_similar_keyword_type = keywords[max_similarity_ind[0]]
        most_similar_keyword = most_similar_keyword_type[max_similarity_ind[1]]
            
        packet = f'{most_similar_keyword},'
        print("Sending:", packet)
        tcpCommunicator.sendToServer(packet)

        print("Sent.")

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

def reset_que():
    prediction_que[:] = []
    tcpCommunicator.command_sent = False
    clear_prediction_que = False

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
mics = [[], []]

prediction_que = []
clear_prediction_que = False
thres = 0.5                         # voice activity threshold
percent_thres = 0.05                # percent of chunks with VA to consider speech

ONE = ["lights", "mute", "unmute"]
TWO = ["lights on", "lights off", "volume down", "volume up"]
THREE = ["overhead lamp off","overhead lamp on", "desk lights off", "desk lights on", "set aux audio", "set phono audio"]
keywords = [THREE, TWO, ONE]
keyword_embeddings = [word_vector_generator.encode(keyword_group) for keyword_group in keywords]
vector_similarity_thres = 0.8
### ----------------------------------------------------------------------------------------

if __name__ == "__main__":    
    print("Waiting to connect to end device...")
    tcpCommunicator.connectClient()
    print("Connected to end device.")

    executor = ThreadPoolExecutor(max_workers=int(longestWhisperBuffer/record_seconds))

    audio_thread = threading.Thread(target=audio_loop)
    audio_thread.start()

    try:
        while running:

            if len(mics[0]) == 0 or len(mics[1]) == 0: continue

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
            number_speech_chunks = torch.sum((VAD_pred > thres).int()).item()
            percent_speech_chunks = number_speech_chunks / VAD_pred.shape[0]

            reached_speech_threshold = percent_speech_chunks > percent_thres
            room_in_que = len(prediction_que) < int(longestWhisperBuffer / record_seconds)

            if reached_speech_threshold and room_in_que:
                prediction_que.append(combined_buffer)
            else:
                clear_prediction_que = True

            del mics[0][0]
            del mics[1][0]

            if len(prediction_que) > 0:
                transcript = prediction(prediction_que)

                if not tcpCommunicator.command_sent:
                    DEBUG = True # if you want to see errors in the threads set to true so the regular threading class is used. 
                    if DEBUG:
                        t = threading.Thread(target=parse_transcript, args=(transcript,), daemon=True)
                        t.start()
                    else:
                        executor.submit(parse_transcript, transcript)

                    if clear_prediction_que:
                        reset_que()
                else:
                    reset_que()
                
    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        executor.shutdown(cancel_futures=True)
        audio_thread.join()
        tcpCommunicator.closeClientConnection()