from ctypes import * 
import numpy as np
from transformers import WhisperForConditionalGeneration, BitsAndBytesConfig
import os
from utils.WhisperProcessor import offlineWhisperProcessor
import torchaudio
from scipy.io.wavfile import write
import wave
import torch
import threading
import timeit
import inspect
from faster_whisper import WhisperModel

os.environ["TRANSFORMERS_OFFLINE"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: {}".format(device))

#quantization_config = BitsAndBytesConfig(load_in_8bit=True)
processor = offlineWhisperProcessor(config_path="utils/preprocessor_config.json",
                                    special_tokens_path="utils/tokenizer_config.json",
                                    vocab_path="utils/vocab.json",
                                    device=device
                                    )
#model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny", local_files_only=True, device_map="auto")
model = WhisperModel("small", device=device, compute_type="float32")
clib = CDLL("./utils/micModule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int)]
clib.accessMicrophone.restype = POINTER(c_short)
clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

"""
ALGORITHIM -----
Run three threads:
    thread 1 (main): running predicition and sending data to Raspberry Pi
    thread 2 (audio): collecting and storing audio buffers
    thread 3 (audio processing): collecting and storing all possible windows then sending them to que for prediction
"""

def audio_loop():
    """
    audio collection loop to collect and store 2 audio buffers
        collected_seconds - the number of seconds of audio to collect from the microphone
    """
    global prev_loc # make variable writeable by thread

    print("Starting audio collection...")
    i = 1
    while running:
        ptr = clib.accessMicrophone(mic_name, sample_rate, channels, buffer_frames, int(record_seconds), byref(sample_count)) # collect mic data
        buffer = np.ctypeslib.as_array(ptr, shape=(sample_count.value,)) # if issues add .copy()
        buffer = buffer.astype(np.float32) / 32768.0 # convert to float32
        clib.freeBuffer(ptr)
 
        # store and track collected audio
        start_ind = prev_loc * buffer_len[0]
        end_ind = prev_loc * buffer_len[0] + sample_count.value 
        
        buffer_storage[start_ind:end_ind] = buffer
        buffer_len[prev_loc] = sample_count.value
        if prev_loc == 1: window_processor_que.append(True) # tell window manager to collect windows

        prev_loc = i % 2
        i += 1

def window_manager():
    """
    collect and manage windows the be predicted on
        window_len - the lenght of the window in frames to be predicted on from the audio buffer
        hop_len - the number of frames to hop forward each window
    """
    global first_frame # make variable writable

    assert window_len > 1 and hop_len <= record_seconds * sample_rate, "The window cannot be longer or shorter than the data collected"
    assert hop_len > 1 and hop_len <= record_seconds * sample_rate, "The window cannot hop farther than the amount of data collected"

    # wait until both buffer locations are filled to start collecting windows
    both_filled = False
    while not both_filled:
        both_filled = buffer_len[0] > 0 and buffer_len[1] > 0
    print("Started window manager...")
    
    while running:
        if len(window_processor_que) > 0 :
            start_ind = first_frame
            end_ind = first_frame + window_len
            
            while end_ind <= len(buffer_storage):
                window_inds = [_ for _ in range(start_ind,end_ind)]
        
                window = buffer_storage[window_inds]
                window_que.append(window)

                first_frame += hop_len
                start_ind = first_frame
                end_ind = first_frame + window_len

            del(window_processor_que[0]) # delete the completed task
        first_frame = 0 # reset first_frame

def prediction():
    #window = torch.from_numpy(window_que[0]).to(device)
    #window = window_que[0]#window.numpy()
    #window = window * (2**15)
    #window = window.astype(dtype=np.int16)
    #wav = wave.open("test_out.wav", "wb")
    #wav.setnchannels(1)
    #wav.setsampwidth(2)  # 16-bit
    #wav.setframerate(16000)
    #wav.writeframes(window)
    #wav.close()
    
    features = processor.extract_features(torch.from_numpy(window_que[0]).to(device))
    features = features.cpu()
    segments = model.transcribe(features, beam_size=5, language="en")
    
   #start_model_gen = timeit.default_timer()
    
    print("===Transcription===")
    for segment in segments:
        print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    #model_pred_time = timeit.default_timer() - start_model_gen 
    #pred = model.generate(input_features=features, language="en")[0]

    #pred = model(input_features=features,
    #             attention_mask=attention_mask,
    #             decoder_input_ids=processor.gen_decoder_ids(),
    #             ).logits[:, -1, :]
    #pred = torch.argmax(pred[0]).item()
    #print(processor.decode_single(pred))
    
    #transcription = []
    #for i, tok in enumerate(pred):
    #    print(tok)
    #    transcription.append(processor.decode_single(tok))

    #transcription = "".join(transcription).replace("Ġ", " ")
    #if transcription[0] == " ":
    #    transcription =  transcription[1:]

    #print(transcription)
    del(window_que[0]) # remove window from que
    print(len(window_que))

    #print("prediction time", timeit.default_timer() - start_pred_loop)
    print("model prediction time", model_pred_time)
        
### SET VARIABLES ###
running = True

sample_count = c_int()
mic_name = b"plughw:CARD=Snowball"
sample_rate = 16000
channels = 1
buffer_frames = 1024
record_seconds = 2.0
window_len = 2 * sample_rate
hop_len = window_len#int((1/16) * sample_rate)

buffer_storage = np.zeros(int(2 * record_seconds * sample_rate), dtype=np.float32) # create storage for two signals
prev_loc = 0 # keep track of last used storage location and initialize
buffer_len = [0, 0] # keep track of how long each buffer is 
window_processor_que = [] # create a que of time the windows should be analyzed

first_frame = 0 # keep track of the first frame for each window
window_que = [] # store windows to be predicted on in a list (queue)

attention_mask = torch.ones(1, 80)
######

audio_thread = threading.Thread(target=audio_loop, daemon=True)
win_thread = threading.Thread(target=window_manager)

audio_thread.start()
win_thread.start()

try:
    while running:
        if len(window_que) > 0:
            prediction()
except KeyboardInterrupt:
    print("Stopping...")
    running = False
    audio_thread.join()
    win_thread.join()

#signal, _ = torchaudio.load("./test_out.wav")
#signal = signal.resize(signal.shape[-1])

#features = processor.extract_features(signal)
#pred = model.generate(features, language="en")[0]

#transcription = []
#for i, tok in enumerate(pred):
#    transcription.append(processor.decode_single(tok))

#transcription = "".join(transcription).replace("Ġ", " ")
#if transcription[0] == " ":
    #transcription =  transcription[1:]

#print("===Transcription===")
#print(transcription)