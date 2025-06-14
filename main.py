import numpy as np
import os
import wave
import torch
import threading
import gensim
import nltk
import gensim.downloader

from faster_whisper import WhisperModel
from multiprocessing import Pool
from utils.LARS_utils import offlineWhisperProcessor, onnxWraper
from ctypes import * 
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize, sent_tokenize

os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

processor = offlineWhisperProcessor(config_path="utils/preprocessor_config.json",
                                    special_tokens_path="utils/tokenizer_config.json",
                                    vocab_path="utils/vocab.json",
                                    device=device
                                    )
vad_model = onnxWraper(".model/silero_vad_16k_op15.onnx", force_cpu=False)
model = WhisperModel("small", device=device, compute_type="float32")
pretrained_vectors = gensim.downloader.load("glove-wiki-gigaword-100")
print("Loaded all models")

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
        transcription.append(segment.text)
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
    parse the transcription and return a list of commands
    The command packet is a byte with a bit field for each command
    There is a byte for related commands with Big Endian format.

    https://www.rapidtables.com/convert/number/hex-to-binary.html?x=03

    bitfield:
        [a, b, c, d, e, _, _, _]
        * if any bits are 0, then there is no command
        a = 1 - turn on lights * if both light commands are 1 then do not send light command
        b = 1 - turn off lights
        c = 1 - mute the sound
        d = 1 - turn volume up
        e = 1 - turn volume down
    """

    # https://techolate.com/how-to-build-a-local-ai-agent-with-python-ollama-langchain-rag/
    # for the transcription, grab three words at a time to create a 3xlen vector
    # using three words at a time to get better context and allow for more than binary commands
    transcription_vectors = []
    transcription = word_tokenize(transcription)
    for word in transcription:
        word = word.lower()
        transcription_vectors.append(pretrained_vectors[word])
    transcription_vectors = np.array(transcription_vectors)
    
    # calculate the cosine similarity between the transcription and each command
    # the matrix created will give the cosine similarity between each transcription and command, where
    #   a row is the similarity between the transcription and each command
    dot_product = np.matmul(transcription_vectors, command_database) # does dot product for each row and column 
    
    transcription_norms = np.linalg.norm(transcription_vectors, axis=1, keepdims=True)
    command_norms = np.linalg.norm(command_database, axis=0, keepdims=True)
    dot_product = dot_product / (transcription_norms * command_norms)
    
    most_similar = np.argmax(dot_product, axis=1) # find the index of the highest cosine similarity

    command_byte1 = 0x00 # initialize the command byte to 0
    for i, ind in enumerate(most_similar):
        if dot_product[i, ind] > similarity_threshold:
            command_byte1 |= 0x01 << ind # set the bit for the command by shifting it ind number of times to the left then adding it to the byte

    # check to make sure the light commands dre not both 1
    lights_on = (command_byte1 & 0x01) == 1
    lights_off = (command_byte1 & 0x02) == 1

    if lights_on and lights_off:
        command_byte1 &= ~0x01 # set both bits to 0

    command_packet.append(int(command_byte1).to_bytes(1, 'big'))

def send_commands():
    # create a byte array of the command packet
    print(command_packet)
    command_packet[:] = [] # clear the command packet

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
energy_threshold = 0.002 # energy threshold to trigger prediction
i = 0

command_packet = []
# make all the commands two words and lower case
command_keywords = ["lights", 
                    "lights", 
                    "mute", 
                    "volume"]
#command_keywords = [word_tokenize(command) for command in command_keywords]

# convert each plain english command to a vector
command_database = [] # store the vector of each corresponding command
print('Generating vector database of commands...')
for keyword in command_keywords:
    command_database.append(pretrained_vectors[keyword])
command_database = np.array(command_database).transpose() # transpose for later matrix multiplication
print('Vector database of commands generated.')
######

if __name__ == "__main__":
    pool = Pool(processes=1)
    
    audio_thread1 = threading.Thread(target=audio_loop, daemon=True)
    audio_thread1.start()

    try:
        while running:
            # check buffer for voice activity
            if len(buffer_que) > 0:
                vad_pred = vad_model(torch.from_numpy(buffer_que[0]))
                num_speech_chunks = torch.sum((vad_pred > thres).int()).item()
                audio_energy = torch.mean(torch.abs(torch.from_numpy(buffer_que[0])))

                if num_speech_chunks > 0 and len(prediction_que) < int(30 / record_seconds) and audio_energy > energy_threshold:
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

                pool.apply_async(parse_prediction, (transcription,), callback=send_commands) # asynchronously parse the prediction and send the commands to the raspberry pi

    except KeyboardInterrupt:
        print("\nStopping...")
        running = False
        audio_thread1.join()
        pool.close()
        pool.terminate()