import pyaudio 
import wave
import glob 
import numpy 
import time
import devices
import random
import os 

CHUNK = 1024
SAMPLE_FORMAT = pyaudio.paInt16
CHANNELS = 1 
FS = 44100
SECONDS = 1

p = pyaudio.PyAudio()
files_recv = False

def find_current_amount_data():
    data = glob.glob('data/*')
    return len(data)

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

amnt_data = find_current_amount_data()

device_index = devices.choose_device('Blue Snowball')
print(f'\n\nThe devce was found..\n')
record_time = float(input('Enter the time between recordings: '))
file_name = input('Enter the prefix for the file names: ')
class_id = int(input('Enter an int representing the class id of the audio: '))

for i in range(get_num_files(files_recv)):
    file_path = f'full_dataset/{class_id}/{file_name + make_file_name()}.wav'
    frames = []

    print(f'******RECORDING STARTING IN {record_time} SECOND******')
    time.sleep(record_time)
    print('*START*\n')
    os.system('mpg123 ./short-beep-tone-47916.mp3')
    time.sleep(0.25)

    stream = p.open(format=SAMPLE_FORMAT,
                    channels=CHANNELS,
                    rate=FS,
                    input=True,
                    input_device_index=int(device_index))
    
    for i in range(0, int(FS / CHUNK * SECONDS)):
        data = stream.read(CHUNK)
        chunk = numpy.frombuffer(data, numpy.int16)
        chunk = chunk * 5 #adjust volume
        frames.append(chunk)
        print ("\033[A                             \033[A")
        print(i)
        
    print('*STOP*\n')
    
    stream.close()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(SAMPLE_FORMAT))
    wf.setframerate(FS)
    wf.writeframes(b''.join(frames))
    wf.close()
    print(f'{file_path} was saved.')

    time.sleep(1)


p.terminate()
