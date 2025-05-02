#include <iostream>
#include <string>

#include <portaudio.h>

#define SAMPLERATE 44100
#define FRAMESPERBUFFER 1024

static void checkErr(PaError err) {
    if (err != paNoError) {
        std::cout << "PortAudio error: " << err << std::endl;
        exit(0);
    }
}

static void printAudioDevices() {
    int numDevices = Pa_GetDeviceCount();
    const PaDeviceInfo* device;
    std::string deviceName;
    int outputChannels;
    int inputChannels;

    std::cout << numDevices << " devices where found..." << std::endl;

    for (int i=0; i<numDevices; i++) {
        device = Pa_GetDeviceInfo(i);
        deviceName = device->name;
        outputChannels = device->maxOutputChannels;
        inputChannels = device->maxInputChannels;

        std::cout << "Device (" << deviceName << ") has: " << outputChannels << "output channels, and " << inputChannels << "input channels\n";
    }
}

static int getMicrophone(std::string name) {
    int numDevices = Pa_GetDeviceCount();
    const PaDeviceInfo* device;
    std::string deviceName;
    int outputChannels;
    int micIndex = -1;

    //find the device index
    for (int i=0; i<numDevices; i++) {
        device = Pa_GetDeviceInfo(i);
        deviceName = device->name;
        outputChannels = device->maxOutputChannels;

        //check if deviceName contains name
        unsigned long int x = 0;
        while (true) {
            if (deviceName[x] == name[x] && x <= name.length() && outputChannels == 0) {
                x++;
            } else { break; }
        }
        if (x == name.length()) { micIndex = i; break; }
    }

    return micIndex;
}

static int callback(
    const void* inputBuffer,
    void* outputBuffer,
    unsigned long framesPerBuffer,
    const PaStreamCallbackTimeInfo* timeInfo,
    PaStreamCallbackFlags statusFlags,
    void* userData
) {
    int* input = (int*)inputBuffer;
    (void)outputBuffer; //prevent unused var error

    return 0;
}

int main() {
    //init portaudio
    PaError err;
    PaStreamParameters inputParameters;
    PaStream* stream;

    err = Pa_Initialize();
    checkErr(err);

    //find mix index
    const std::string name = "Blue Snowball";
    int micIndex = getMicrophone(name);

    if (micIndex < 0) {
        std::cout << "The device (" << name << ") could not be found. \n";
        std::cout << "This is a list of all the devices:\n";
        printAudioDevices();
        return 0;
    }
    std::cout << "===========================================" << std::endl;
    std::cout << "The device (" << name << ") was found at index: " << micIndex << std::endl;

    err = Pa_OpenStream(
        &stream,
        &inputParameters,
        NULL,
        SAMPLERATE,
        FRAMESPERBUFFER,
        paNoFlag,
        callback,
        NULL
    );
    checkErr(err);
    std::cout << "Opened stream." << std::endl;

    err = Pa_StartStream(&stream);
    checkErr(err);
    std::cout << "Started stream." << std::endl;

    err = Pa_StopStream(&stream);
    checkErr(err);
    std::cout << "Stream stopped." << std::endl;

    err = Pa_Terminate();
    checkErr(err);
    return 0;
}