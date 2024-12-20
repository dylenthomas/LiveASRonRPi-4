import wave
import pyaudio
import os
from audio_eval import get_input, CHUNK

p = pyaudio.PyAudio()

def main():
    i = 1
    while True:
        audio_files = open('./double-check.txt').read().split('\n')
        if len(audio_files[0]) < 1: break 
        
        print(i)
        playback(audio_files[0])
        
        #remove the file just read 
        with open('./double-check.txt', 'w') as file:
            data = '\n'.join(audio_files[1:])
            file.write(data)
            
        i += 1
         
    
def playback(abs_dir):
    abs_dir_split = abs_dir.split('/')
    file = abs_dir_split[len(abs_dir_split) - 1]
     
    print(f'Playing: {abs_dir}')
     
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
        class_dir = '/'.join(abs_dir_split[:-1])
        class_dir = class_dir[:-1] + new_class #replace the class dir with the new one
        os.rename(abs_dir, os.path.join(class_dir, file)) 
        abs_dir = os.path.join(class_dir, file)
         
    wf.close()
    stream.close()
    
    
if __name__ == '__main__':
    main()