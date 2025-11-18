import numpy as np
import sys
import socket
import torch

from sentence_transformers import SentenceTransformer, util

class kwVectorHelper():
    def __init__(self, device, similarity_threshold):
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
        self.keywords = [THREE, TWO, ONE]

        self.vector_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_model.to(device)
        self.keyword_embeddings = [self.vector_model.encode(THREE), self.vector_model.encode(TWO), self.vector_model.encode(ONE)]

        self.similarity_threshold = similarity_threshold

    def parse_prediction(self, transcription, tcpCommunicator):
        """
        parse the transcript and use matrix/vector math to find words/pharases which are similar to the desired command keywords
        """ 

        transcription = [" ".join(transcription)]
        transcript_embedding = self.vector_model.encode(transcription)

        max_similarity = 0.0
        max_similarity_ind = (0, 0)

        for i in range(len(self.keyword_embeddings)):
            kw_matrix = self.keyword_embeddings[i]

            similarities = util.cos_sim(transcript_embedding, kw_matrix)
            most_similar = torch.argmax(similarities)
            
            if similarities[0, most_similar] > max_similarity:
                max_similarity = similarities[0, most_similar]
                max_similarity_ind = (i, most_similar.item())

        if max_similarity > self.similarity_threshold:
            most_similar_keyword_type = self.keywords[max_similarity_ind[0]]
            most_similar_keyword = most_similar_keyword_type[max_similarity_ind[1]]
            self.send_commands([most_similar_keyword], tcpCommunicator)

    def send_commands(self, found_keywords, tcpCommunicator):
        """
        Create a command packet with the keywords found in the transcription 
        The command packet is created with command encodings where the command is encoded as its index in the list

        https://www.rapidtables.com/convert/number/hex-to-binary.html?x=03
        """
        packet = ''
        for i in found_keywords:
            packet += str(i) + ',' # seperate each command by a comma
            print("Sent: ", packet)
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