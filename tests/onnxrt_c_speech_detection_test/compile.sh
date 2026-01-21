#!/bin/bash

ONNXRUNTIME_ROOT="/home/dylenthomas/Calliope/LARS/onnxruntime-linux-x64-1.12.1"
PROJECT_ROOT="/home/dylenthomas/Calliope/LARS/LiveASRonRPi-4"

gcc -Wall onnxrt_c_speech_detection_test.c -I $ONNXRUNTIME_ROOT/include -I $PROJECT_ROOT/include \
	-L $ONNXRUNTIME_ROOT/lib -L $PROJECT_ROOT/libs -lasound -lonnxruntime -lmic_access \
	-Wl-rpath="$ONNXRUNTIME_ROOT/lib" -o test 
