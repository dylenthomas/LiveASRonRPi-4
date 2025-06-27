transcription = ["Turn", "off", "the", "lights", "and", "open", "the", "blinds"]

transcrpt_vecs = []
for i in range(3):
    i = 3 - i # count down from longest to shortest to match the keyword formatting

    vec = []
    for w in range(len(transcription) - (i - 1)):
        words = [word.lower() for word in transcription[w : w + i]]
        print(words)
        #vec.append(pretrained_vectors[words])
        vec.append(words)
    #vec = np.array(vec)
    print("====")
    transcrpt_vecs.append(vec)

print(transcrpt_vecs)
