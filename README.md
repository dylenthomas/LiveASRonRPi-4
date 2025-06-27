# LARS
### Live Asr with a Raspberry-pi as the end device and a dGPU on a Server doing all the heavy lifting (LARS)

As the acronym suggests I am running real time ASR on a dedicated server node in my house to allow for home automation, with a raspberry pi as the end device to do the "automation stuff"

This current version is using a modified version of [faster-whisper](https://github.com/SYSTRAN/faster-whisper) which is in the utils folder. The only changes made were those so that a processed audio buffer can be set for prediction instead of it expecting an audio file. Because of the speed of faster-whisper I can get instant and accurate transcriptions so thank you to them for their amazing work. 