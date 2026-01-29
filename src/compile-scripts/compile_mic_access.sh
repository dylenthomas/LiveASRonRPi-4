#!/bin/bash
gcc -shared -o ./libs/libmic_access.so \
	-fPIC mic_access.c \
	-Wall \
	-I ./include/ \
	-lasound \
