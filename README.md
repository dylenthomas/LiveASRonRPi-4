# LARS
### Live Asr with a Raspberry-pi as the end device and a dGPU on a Server doing all the heavy lifting (LARS)

As the acronym suggests this repo is for running real time ASR on a dedicated server node and communicating the transcribed audio with a raspberry-pi for home automation

This current version is using a modified version of [faster-whisper v1.1.1](https://github.com/SYSTRAN/faster-whisper) which is in the utils folder. The only changes made were those so that a processed audio buffer can be set for prediction instead of it expecting an audio file. Because of the speed of faster-whisper I can get instant and accurate transcriptions so thank you to them for their amazing work. I am also using an onnx model from [silero-vad](https://github.com/snakers4/silero-vad) which works extremely well, so thank you to them. 