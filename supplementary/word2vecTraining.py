import json
from gensim.models import Word2Vec

### Open and process the whisper corpus ----
with open("WhisperCorpus.json", "r") as jsn_data:
    corpus = jsn_data.read()
    jsn_data.close()

# remove the first colon
corpus = corpus.split(',')
corpus[0] = corpus[0].replace("{", "")

# remove Gs and indexes
for i, samp in enumerate(corpus):
    if "Ġ" in samp: samp = samp.replace("Ġ", "")
    samp = samp.replace('"', "")
    colon = samp.find(":")
    samp = samp[0:colon]
    corpus[i] = samp

# rejoin into a string 
corpus = ", ".join(corpus)

# write into text file
#with open("processedWhisperCorpus.txt", "w") as processed:
#    processed.write(corpus)
#    processed.close()