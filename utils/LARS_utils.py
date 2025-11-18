import numpy as np
import sys
import socket

from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize

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

    def parse_prediction(self, transcription, tcpCommunicator):
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

        self.send_commands(found_keywords, tcpCommunicator)

    def send_commands(self, found_keywords, tcpCommunicator):
        """
        Create a command packet with the keywords found in the transcription 
        The command packet is created with command encodings where the command is encoded as its index in the list

        https://www.rapidtables.com/convert/number/hex-to-binary.html?x=03
        """
        packet = ''
        for i in found_keywords:
            packet += str(i) + ',' # seperate each command by a comma
        if len(packet) == 0: return # no keywords

        tcpCommunicator.sendToServer(packet)

class TCPCommunication():
    def __init__(self):
        self.ip = "100.72.193.15"
        self.port = 5000
        self.buff_size = 1024
        self.command_sent = False

    def openServer(self):
        maxConnections = 1

        print("Waiting for connection...")
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # prevent address already in use error
            try:
                s.bind((self.ip, self.port))
                s.listen(maxConnections)
            
                conn, addr = s.accept()
            except socket.error as msg:
                print("[TCP ERROR]: {}".format(msg))
                sys.exit(1)
            self.s = s

        self.conn = conn
        self.addr = addr
        print("Found connection.")

    def readFromClient(self):
        return self.conn.recv(self.buff_size).decode("utf-8")

    def connectClient(self):
        try:
            self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.s.connect((self.ip, self.port))
        except socket.error as msg:
            print("[TCP ERROR]: {}".format(msg))
            sys.exit(2)

    def sendToServer(self, data):
        data = data.encode("utf-8")
        if not self.command_sent:
            self.s.send(data)
            self.command_sent = True

    def closeClientConnection(self):
        self.s.close()