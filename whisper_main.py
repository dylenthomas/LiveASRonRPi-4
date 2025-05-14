from ctypes import * 

import tensorflow as tf
import librosa
import os
import numpy as np

ASR_dir = "/home/dylenthomas/whisperProject/ASR_TFLite" 
model_path = f"{ASR_dir}/pre_trained_models/English/subword-conformer.latest_for_english.tflite"
audio_path = f"{ASR_dir}/wavs/1089-134691-0000.flac"

audio_data, _ = librosa.load(os.path.expanduser(audio), sr=16000, mono=True)
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.resize_tensor_input(input_details[0]["index"], signal.shape)
interpreter.allocate_tensors()
interpreter.set_tensor(input_details[0]["index"], signal)
interpreter.set_tensor(
    input_details[1]["index"],
    np.array(0).astype("int32")
)
interpreter.set_tensor(
    input_details[2]["index"],
    np.zeros([1,2,1,320]).astype("float32")
)
interpreter.invoke()
pred = interpreter.get_tensor(output_details[0]["index"])

print("".join([chr(u) for u in pred]))

exit()
#class GreedyCTCDecoder(torch.nn.Module):
#    def __init__(self, labels, blank=0):
#        super().__init__()
#        self.labels = labels
#        self.blank = blank
#
#    def forward(self, emission: torch.Tensor):
#        indices = torch.argmax(emission, dim=-1) #[num, seq]
#        indices = torch.unique_consecutive(indices, dim=-1)
#        indices = [i for i in indices if i != self.blank]
#        return "".join([self.labels[i] for i in indices])
#
#labels = ('-', '|', 'E', 'T', 'A', 'O', 'N', 'I', 'H', 'S', 'R', 'D', 'L', 'U', 'M', 'W', 'C', 'F', 'G', 'Y', 'P', 'B', 'V', 'K', "'", 'X', 'J', 'Q', 'Z')

clib = CDLL("/home/dylenthomas/whisper/whispermodule.so")

#define c++ functions
clib.accessMicrophone.argtypes = [c_char_p, c_uint, c_int, c_int, c_int, POINTER(c_int)]
clib.accessMicrophone.restype = POINTER(c_short)

clib.freeBuffer.argtypes = [POINTER(c_short)]
clib.freeBuffer.restype = None

sample_count = c_int()
ptr = clib.accessMicrophone(b"microphone name", 16000, 1, 512, 5, byref(sample_count))
mic_samples = [ptr[i] for i in range(sample_count.value)]
clib.freeBuffer(ptr)

#Wav2Vec2 = torch.jit.load("/home/dylenthomas/whisper/.model/Wav2Vec2.pt")
#decoder = GreedyCTCDecoder(labels)

#mic_samples = torch.tensor(mic_samples)
#with torch.inference_mode():
#    emission, _ = Wav2Vec2(mic_samples)

#transcript = decoder(emission[0])
#print(transcript)
