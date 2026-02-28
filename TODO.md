 - [x] Get VAD working on GPU
    Prediction runs at ~ 500ms (*I think*, I got my time measurement code from Gemini...) 
 - [x] Try adding peak-hold algorithim
    Try and mimic peak-hold circuit behavoir on the predicted value
    Currently I feel like its moving *too fast* when it resets every iteration
 - [x] Incorporate both microphones
 - [x] Setup threading for each microphone
 - [x] Figure out how to combine microphones into single stream
 - [ ] Combine microphones into single stream
 - [ ] Get python code that runs Whisper working in concert with C code
 - [ ] Combine pieces for a C based transcripter
 - [ ] Roll TCP communication
