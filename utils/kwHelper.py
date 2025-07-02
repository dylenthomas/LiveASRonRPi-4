import gensim.downloader
import numpy as np

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

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

class kwVectorHelper:
    def __init__(self):
        self.vector_model = gensim.downloader.load("glove-wiki-gigaword-100")
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

    def get_kw_mat(self):
        return self.vecs

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