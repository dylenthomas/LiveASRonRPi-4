import pyaudio
import numpy as np
import torch 
import torchaudio
import time

class wake_word_listener():
    def __init__(self, model_path: str, format=pyaudio.paInt16, rate=44100):
        self.rate = rate
        self.p = pyaudio.PyAudio()
        self.device_index = self.get_device_index()
        self.stream = self.p.open(format=format,
                                  channels=1,
                                  rate=rate,
                                  input=True,
                                  frames_per_buffer=rate,
                                  input_device_index=self.device_index,
                                  stream_callback=self.callback)
        
        self.model = torch.jit.load(model_path).eval()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=rate,
            n_fft=1024,
            hop_length=512,
            n_mels=64
        )
        self.last_timestamp = time.time()
        
    def get_device_index(self):
        """
            Find the device index corresponding to Blue Snowball'
        """
        num_devices = self.p.get_device_count()

        for i in range(num_devices):
            device_info = self.p.get_device_info_by_index(i)
            
            if 'Blue Snowball' in device_info.get('name'):
                device_index = i
        try:
            return device_index
        except UnboundLocalError:
            raise RuntimeError ('Device not found.')
    
    def callback(self, in_data, frame_count, time_info, status):
        """
            Callback function for non-blocking opperation:
            -->(called in a seperate thread from the main thread)

            in_data: input data
            frame_count: number of frames
            time_info: dictionary with time info
            status: PaCallbackFlags
        """

        np_data = np.frombuffer(in_data, dtype=np.int16)
        #np_data = np_data * 5 #boost the volume slightly
        self.audio_data = (np_data, time.time())

        return (np_data, pyaudio.paContinue)
    
    def predict(self):
        try:
            self.audio_data[0][0]
        except AttributeError:
            return None #audio data hasn't been assigned yet

        tensor_data = torch.tensor(self.audio_data[0]).to(dtype=torch.float)
        tensor_data = torch.FloatTensor(tensor_data)
        tensor_data = torch.reshape(tensor_data, (1, tensor_data.shape[0])) #reshape for channel num
        tensor_data = torch.reshape(tensor_data, (1, tensor_data.shape[0], tensor_data.shape[1]))
        tensor_data = self.mel_transform(tensor_data)

        if self.audio_data[1] != self.last_timestamp:
            with torch.no_grad():
                predictions = self.model(tensor_data)
                predicted_indx = predictions[0].argmax(-1)
                predicted = predicted_indx

            self.last_timestamp = self.audio_data[1]
        
            return predicted.item(), predictions
        
        else:
            return None
    
if __name__ == '__main__':
    listener = wake_word_listener('whisper.pt')
    listener.stream.start_stream()
    
    try:
        while listener.stream.is_active():
            prediction = listener.predict()
            if prediction is not None:
                predicted, tensor = prediction
                if predicted == 1:
                    print('Hello.')
                else:
                    print('...')

    except KeyboardInterrupt:
        listener.stream.stop_stream()
        listener.stream.close()