import pyaudio 
import wave
import numpy as np 
import time
import devices
import random
import os 

CHUNK = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1 
FS = 48000
SECONDS = 1

p = pyaudio.PyAudio()
files_recv = False

"""
0 - background-noise
1 - talking
2 - music
3 - target-whistle
4 - person-in-room 
"""

def get_num_files(files_recv):
    while not files_recv:
        num_files = input('Enter the amount of files you want to make:')

        try:
            num_files = int(num_files)
            files_recv = True
        except:
            print('Enter a number.')
    
    return num_files

def make_file_name():
    options = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz12345678910'
    file_name = ''
    
    for i in range(5):
        choice = random.randrange(0, len(options) - 1)
        file_name += options[choice]

    return '_' + file_name

device_index = devices.choose_device('Blue Snowball')
print(f'\n\nThe devce was found..\n')

record_time = float(input('Enter the time between recordings: '))
file_name = input('Enter the prefix for the file names: ')
class_id = int(input('Enter an int representing the class id of the audio: '))

for i in range(get_num_files(files_recv)):
    file_path = f'full_dataset/{class_id}/{file_name + make_file_name()}.wav'
    frames = []
    
    is_stream_listening = False

    stream = p.open(format=SAMPLE_FORMAT,
                    channels=CHANNELS,
                    rate=FS,
                    input=True,
                    input_device_index=int(device_index)
                    )
    
    print(f'******RECORDING STARTING IN {record_time} SECOND******')
    time.sleep(record_time)
    print('*START*\n')
    os.system('mpg123 ./short-beep-tone-47916.mp3')
    print()
    time.sleep(0.25) #make sure the mic doesn't capture the beep
    
    for i in range(0, round(FS/CHUNK * SECONDS) + 1):
        chunk = stream.read(CHUNK, exception_on_overflow=False)
        np_data = np.frombuffer(chunk, np.int16)
        frames.append(np_data)
        print ("\033[A                             \033[A")
        print(i)
    print('*STOP*\n')
    
    
    stream.close()
   
    if file_name == 'target-whistle':
        while True:
            delete = input('Would you like to keep the file? (y/n)\n')
            if delete != 'y' or delete != 'n':
                print('enter y/n')
            else:
                if delete == 'y':
                    delete = True
                    break 
                else:
                    delete = False
                    break 
    else:
        delete = False
         
    if not delete: 
        wf = wave.open(file_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
        wf.setframerate(FS)
        wf.writeframes(b''.join(frames))
        wf.close()
        print(f'{file_path} was saved.')

    time.sleep(1)

p.terminate()