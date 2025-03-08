import os
import copy

DATA_DIR = "/Volumes/EXTREME SSD/LibriSpeech"

"""
Create a dataset with the wav data and a transcript for each file
"""

def move_wav(dirpath, filename):
    #directory bs
    old_dirpath = dirpath
    new_dirpath = copy.deepcopy(dirpath)
    new_dirpath = new_dirpath.replace(DATA_DIR, '')
    new_dirpath = os.path.normpath(f'{DATA_DIR}WAV'+new_dirpath)
    
    #make new directory if needed
    if not os.path.exists(new_dirpath): os.makedirs(new_dirpath)
    
    #Move the wav file
    os.rename(os.path.join(old_dirpath, filename), os.path.join(new_dirpath, filename))
    
def extract_transcript(dirpath, filename):
    #directory bs
    old_dirpath = dirpath
    new_dirpath = copy.deepcopy(dirpath)
    new_dirpath = new_dirpath.replace(DATA_DIR, '')
    new_dirpath = os.path.normpath(f'{DATA_DIR}WAV'+new_dirpath)
    
    #make new directory if needed
    if not os.path.exists(new_dirpath): os.makedirs(new_dirpath)    

    #get transcript into list
    with open(os.path.join(old_dirpath, filename), 'r') as trans:
        transcript = trans.read().split('\n')
        trans.close()
        
    #seperate filename and transcript data
    new_transcript = {}
    for trial in transcript:
        space_split = trial.split(' ')
        trial_name = space_split[0]
        if trial_name != '': new_transcript[trial_name] = trial.replace(trial_name, '')
    
    keys = list(new_transcript.keys())
    for key in keys:
        with open(os.path.join(new_dirpath, key+'.txt'), 'w') as trans:
            trans.write(new_transcript[key])
            trans.close()

for dirpath, _, filenames in os.walk(DATA_DIR):
    for filename in filenames:
        if filename.endswith('.wav'):
            move_wav(dirpath, filename)
        elif filename.endswith('.trans.txt'):
            extract_transcript(dirpath, filename)