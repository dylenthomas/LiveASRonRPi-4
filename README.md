# whisper

Speech To Text (STT) system for controlling systems in my room.

### To-Do

* [X] Seperate the transcripts into individual files that correspond to each wav file
* [X] Move the wav files and the individual transcripts to a seperate dataset
* [X] Generate TextGrid alignment files
* [X] Use time stamps in alignment files to create continuous labels
* [X] Create code to convert a TextGrid to encoded continuous ground truth data
* [X] Make dataloader
* [X] Figure out why dataloader uses almost 100gb of memory when using more than 1 input indx
* [X] Make training code
* [ ] Figure out why the loss is outrageously high
