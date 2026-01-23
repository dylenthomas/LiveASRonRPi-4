#!/bin/bash

PROJECT_ROOT="/home/dylenthomas/Calliope/LARS/LiveASRonRPi-4"

gcc -Wall onnxrt_c_speech_detection_test.c \
	-I $PROJECT_ROOT/include \
	-L $PROJECT_ROOT/libs \
	-lasound -lonnxruntime -lmic_access \
	-o test 
