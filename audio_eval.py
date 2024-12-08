import pyaudio
import wave
import argparse
import os

DATSET_DIR = "/Users/dylenthomas/Documents/whisper/full_dataset/"
CHUNK = 1024

#init parser
parser = argparse.ArgumentParser()
parser.add_argument('class_int', help='The int of the class id to iterate through', type=int)
args = parser.parse_args()

#init pyaudio
p = pyaudio.PyAudio()

def get_input(message, *args):
    print(message) 
    cond = True
    while cond:
        wrong = 0
        choice = input(': ')
        for _ in args:
            if choice == _:
                cond = False
                break  
            else:
                wrong += 1

        if wrong == len(args):
            print(f'{choice}, is not one of the available options: {args}')
            print('Please try again')    

    return choice

def playback(file, class_dir):
    checked = open('/Users/dylenthomas/Documents/whisper/checked.txt')
    add_checked = open('/Users/dylenthomas/Documents/whisper/checked.txt', 'a')

    abs_dir = os.path.join(class_dir, file) 
    if abs_dir in checked.read().split('\n'):
        checked.close()
        add_checked.close() 
        return
    
    print(f'Playing: {file}')
     
    wf = wave.open(abs_dir, 'rb')
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                    )
    while True:
        data = wf.readframes(CHUNK)
        while data:
            stream.write(data)
            data = wf.readframes(CHUNK)
         
        if get_input('Repeat? (y/n)', 'y', 'n') == 'n':
            break
        else:
            wf.rewind() #reset pointer to start of file
            
    if get_input('Move the file? (y/n)', 'y', 'n') == 'y':
        new_class = get_input('What class do you want to move the file too?', '0', '1', '2', '3', '4', '5')
        class_dir = class_dir[:-1] + new_class #replace the class dir with the new one
        os.rename(abs_dir, os.path.join(class_dir, file)) 
        abs_dir = os.path.join(class_dir, file)
         
    add_checked.write(abs_dir + '\n')
    checked.close()
    add_checked.close() 
    wf.close()
    stream.close()
    
def main():
    print(f'Evaluating class {args.class_int}')
   
    class_dir = os.path.join(DATSET_DIR, str(args.class_int))  
    data = os.listdir(class_dir)
    print(f'There are {len(data)} files in this class')
    
    if get_input('Do you want to continue? (y/n)', 'y', 'n') == 'n':
        return
    
    print('Starting playback.')
   
    for file in data:
        playback(file, class_dir) 
         
    
     
if __name__ == '__main__':
    main()