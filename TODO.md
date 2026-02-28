 - [x] Get VAD working on GPU
    Prediction runs at ~ 500ms (*I think*, I got my time measurement code from Gemini...) 
 - [x] Try adding peak-hold algorithim
    Try and mimic peak-hold circuit behavoir on the predicted value
    Currently I feel like its moving *too fast* when it resets every iteration
 - [x] Incorporate both microphones
 - [ ] Setup threading for each microphone
 - [ ] Figure out how to combine microphones into single stream
    Use Delay-Sum Beamforming (http://www.labbookpages.co.uk/audio/beamforming/delayCalc.html)
 - [ ] Get python code that runs Whisper working in concert with C code
 - [ ] Combine pieces for a C based transcripter
 - [ ] Roll TCP communication
