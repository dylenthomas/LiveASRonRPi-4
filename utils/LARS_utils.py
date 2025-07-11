import json
import torchaudio.transforms as T
import torch
import torch.nn as nn
import librosa
import onnxruntime as ort
import numpy as np

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

class LogMelSpectrogram():
    """Compute the log mel spectrogram of an audio waveform
    
        Audio must be sampled at 16,000 sr
        Audio must also be mono
    """
    def __init__(self, mel_filters:torch.Tensor, sample_rate:int, n_samples:int, n_fft:int, hop_length:int, device:str="cpu"):
        self.mel_floor = 1e-10

        self.mel_filters = mel_filters
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.device = device

        self.spectrogram_transform = T.Spectrogram(n_fft=n_fft,
                                                   hop_length=hop_length,
                                                   power=2.0,
                                                   ).to(self.device)
        self.db_transform = T.AmplitudeToDB(stype="power").to(self.device)

    def convert_to_mel_scale(self, spectrogram):
        return torch.maximum(torch.Tensor([[self.mel_floor]]).to(device=self.device), torch.mm(self.mel_filters, spectrogram))

    def pad_or_trim_waveform(self, waveform):
        waveform_length = waveform.shape[-1]
        if waveform_length > self.n_samples:
            waveform = waveform[:-(waveform_length - self.n_samples)]
        elif waveform_length < self.n_samples:
            pad = (0, self.n_samples - waveform_length)
            waveform = nn.functional.pad(waveform, pad, "constant", value=0)
        return waveform
   
    def compute(self, x):
        #x = self.pad_or_trim_waveform(x)
        x = self.spectrogram_transform(x)
        x = self.convert_to_mel_scale(x)

        # Got the rest from this function vvvvvv
        # /home/dylenthomas/miniconda3/envs/python310/lib/python3.10/site-packages/transformers/models/whisper/feature_extraction_whisper.WhisperFeatureExtractor._np_extract_fbank_features
        # not sure why they did it this way but this makes the spectrogram the same as their feature extractor

        x = torch.abs(x)
        x = torch.log10(x) # convert to log scale
        x = torch.maximum(x, torch.max(x) - 8.0)
        x = (x + 4.0) / 4.0
        
        return x

     
class offlineWhisperProcessor():
    """Offline processor for whisper decoding and feature extraction
    """
    def __init__(self, config_path:str, special_tokens_path:str, vocab_path:str, device:str="cpu"):
        with open(config_path, 'r') as a:
            self.config = json.load(a)
            a.close()

        with open(special_tokens_path, 'r') as b:
            self.special_tokens = json.load(b)["added_tokens_decoder"]
            b.close()

        with open(vocab_path, 'r') as c:
            self.vocab = json.load(c)
            c.close()

        self.device = device

        self.sample_rate = self.config["sampling_rate"]
        self.hop_length = self.config["hop_length"]
        self.mel_filters = torch.Tensor(self.config["mel_filters"]).to(self.device)
        self.n_fft = self.config["n_fft"]
        self.n_samples = self.config["n_samples"]
        self.nb_max_frames = self.config["nb_max_frames"]

        self.lm_spect_transform = LogMelSpectrogram(self.mel_filters, self.sample_rate, self.n_samples, self.n_fft, self.hop_length, device=self.device)
    
    def decode_single(self, val):
        if val > 50257:
            return self.special_tokens[str(val)]["content"]
        else:
            return list(self.vocab.keys())[val] # val corresponds to the index

    def print_config(self):
        for key in self.config.keys():
            print(key)
            print("\t", self.config[key])

    def extract_features(self, waveform):
        features = self.lm_spect_transform.compute(waveform)
        if features.shape[-1] > self.nb_max_frames: # sometimes is 3001 and not 3000
            features = features[:, :-(features.shape[-1] - self.nb_max_frames)]
        return features
    
    def gen_decoder_ids(self):
        # <|startoftranscript|> <|en|> <|transcribe|> <|notimestamps|> 
        ids = [50258, 50259, 50359, 50363]
        return torch.tensor([ids], dtype=torch.long).to(self.device)

class onnxWraper():
    """
    Onnx wrapper for a VAD model, this model gives a percentage of certainty that there is human speech in each chunk of audio provided.
    The chunk length should be 512 for 16kHz audio, so with shape (10, 512) each of those 10 chunks will get a percentage chance that there is human speech in that audio.
    """
    def __init__(self, path, force_cpu=False):
        ort_opts = ort.SessionOptions()
        ort_opts.inter_op_num_threads = 1
        ort_opts.intra_op_num_threads = 1

        if force_cpu and "CPUExecutionProvider" in ort.get_available_providers():
            self.inference_ses = ort.InferenceSession(path, providers=["CPUExecutionProvider"], sess_options=ort_opts)
            print("onnx is using CPU")
        else:
            self.inference_ses = ort.InferenceSession(path, providers=["CUDAExecutionProvider"], sess_options=ort_opts)
            print("onnx is using CUDA")

        self._reset_states()

    def _validate_shape(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        elif x.dim() >  2:
            raise ValueError(f"Too many dimensions for input audio chunk {x.dim()}")

        return x

    def _reset_states(self, batch_size=1):
        self._state = torch.zeros((2, batch_size, 128)).float()
        self._context = torch.zeros(0)
        self._last_batch_size = 0

    def __call__(self, x):
        x = self._validate_shape(x)
        num_samples = 512
        if x.shape[-1] != num_samples and x.shape[-1] % num_samples != 0:
            # reshape the data so the VAD model can process it
            remainder = x.shape[-1] % num_samples
            
            remainder_data = x[:, -remainder:] # set aside the remainder data
            x = x[:, :-remainder] # cut off the remainder data

            x = torch.reshape(x, (x.shape[-1] // num_samples, num_samples)) # reshape data
            remainder_data = nn.functional.pad(remainder_data, (0, num_samples-remainder), "constant", 0) # zero pad

            x = torch.cat((x, remainder_data))
        
        else:
            x = torch.reshape(x, (x.shape[-1] // num_samples, num_samples))
            
        batch_size = x.shape[0]
        context_size = 64

        if not self._last_batch_size:
            self._reset_states(batch_size)
        if self._last_batch_size and self._last_batch_size != batch_size:
            self._reset_states(batch_size)

        if not len(self._context):
            self._context = torch.zeros(batch_size, context_size)

        x = torch.cat([self._context, x], dim=1)

        ort_inputs = {"input": x.numpy(), "state": self._state.numpy(), "sr": np.array(16000, dtype="int64")}
        ort_outs = self.inference_ses.run(None, ort_inputs)
        out, state = ort_outs
        self._state = torch.from_numpy(state)

        self._context = x[..., -context_size:]
        self._last_batch_size = batch_size

        out = torch.from_numpy(out)
        return out

class kwVectorHelper():
    def __init__(self):
        ONE = ["lights",
            "mute",
            "unmute"]
        TWO = ["lights on",
            "lights off",
            "volume down",
            "volume up",]
        THREE = ["overhead lamp off",
                "overhead lamp on",
                "desk lights off",
                "desk lights on",
                "set aux audio",
                "set phono audio"]

        TWO= [word_tokenize(kw) for kw in TWO]
        THREE = [word_tokenize(kw) for kw in THREE]

        #self.vector_model = gensim.downloader.load("glove-wiki-gigaword-100")
        self.vector_model = KeyedVectors.load("/home/dylenthomas/LiveASRonRPi-4/.model/LARS.wordvectors")
        self.vecs = [THREE, TWO, ONE] # check from longest to shortest to avoid scenarios where a shorter keyword that lies in a longer one is identified as the indended keyword
        self.encodings = {}

        kw_ind = 0
        for i, kw_list in enumerate(self.vecs):
            db = []
            for kw in kw_list:
                if i == len(self.vecs) - 1: # one word keywords
                    db.append(self.vector_model[kw])
                    kw = [kw] # to prevent running .join on a string instead of a list below
                else:
                    db.append(np.concatenate(self.vector_model[kw]))

                # encode keywords as indexes
                self.encodings[" ".join(kw)] = kw_ind
                kw_ind += 1

            self.vecs[i] = np.array(db).transpose()

    def get_encodings(self):
        return self.encodings

    def transcript2mat(self, transcription):
        transcrpt_vecs = []
        for i in range(len(self.vecs)):
            i = len(self.vecs) - i # count down from longest to shortest to match the keyword formatting

            vec = []
            for w in range(len(transcription) - (i - 1)):
                words = [word.lower() for word in transcription[w : w + i]] # make everything lower case
                if i == 1:
                    vec.append(self.vector_model[words[0]])
                else:
                    vec.append(np.concatenate(self.vector_model[words]))
            vec = np.array(vec)
            transcrpt_vecs.append(vec)

        return transcrpt_vecs

    def setThreadManager(self, thread_executor):
        self.executor = thread_executor

    def addParsingThread(self, transcription):
        self.executor.submit(self.parse_prediction, transcription)

    def parse_prediction(self, transcription):
        """
        parse the transcript and use matrix/vector math to find words/pharases which are similar to the desired command keywords
        """ 
        similarity_threshold = 0.8

        transcription = " ".join(transcription)
        transcription = word_tokenize(transcription)

        # make a list of lists to get multiple word chunks that line up with the comand keyword vectors
        transcrpt_vecs = self.transcript2mat(transcription)

        # calculate the cosine similarity between the transcription and each keyword 
        # the matrix created will give the cosine similarity between each transcription and keyword, where
        #   a row is the similarity between a given chunk of the transcript and each keyword 
        found_keywords = []
        for i, kw_matrix in enumerate(self.vecs):
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
                found_keywords.append(self.encodings[found_kw])

        self.send_commands(found_keywords)

    def send_commands(self, found_keywords):
        """
        Create a command packet with the keywords found in the transcription 
        The command packet is created with command encodings where the command is encoded as its index in the list

        https://www.rapidtables.com/convert/number/hex-to-binary.html?x=03
        """
        packet = ''
        for i in found_keywords:
            packet += str(i) + ',' # seperate each command by a comma
        if len(packet) == 0: return # no keywords
        print(packet)

        # send the data as a character array of bytes
        #sent_bytes = clib_serial.writeSerial(serial, packet.encode('utf-8'), len(packet))

        #if sent_bytes == -1:
        #    print("There was error writing to serial.")
        #else:
        #    print("{} bytes where sent through serial".format(sent_bytes))
        #    commands_sent = True

class TCPCommunication():
    def __init__(self):
        self.ip = "100.72.193.15"
        self.port = 5000

        self.buff_size = 1024

    def connectServer(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.ip, self.port))
            s.listen(1)

            conn, addr = s.accept()

        self.conn = conn
        self.addr = addr

    def readFromClient(self):
        return self.conn.recv(self.buff_size)

    def connectClient(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.ip, self.port))

    def sendToServer(self, data):
        self.s.send(data)
        return s.recv(self.client_buff_size)

    def closeClientConnection(self):
        self.s.close()
