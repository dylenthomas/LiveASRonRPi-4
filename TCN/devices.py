import pyaudio

def choose_device(device_name):
   p = pyaudio.PyAudio()

   devices = p.get_device_count()

   for i in range(devices):
      device_info = p.get_device_info_by_index(i)
      print(device_info.get('index'))
      print(device_info.get('name'))

      if device_info.get('maxInputChannels') > 0:
         if device_name in device_info.get('name'):
            device_index = i

   try:      
      return device_index
   except UnboundLocalError:
      raise RuntimeError ('Microphone not found.')

if __name__ == '__main__':
   choose_device('Blue Snowball')