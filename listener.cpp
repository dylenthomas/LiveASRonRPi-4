#include <string>
#include <iostream>
#include <memory>
#include <chrono>
#include <thread>
#include <cstring>

#include <torch/script.h>
#include <portaudio.h>

#define SAMPLERATE 44100
#define FRAMESPERBUFFER 44100 //make each buffer hold 1 second of data for prediction

//https://pytorch.org/tutorials/advanced/cpp_export.html
//https://github.com/pytorch/examples/tree/main/cpp

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

class Listener {
    private:
        PaStreamParameters inputParameters;
        PaError err;
        PaStream* stream;

        torch::jit::script::Module model;
        std::vector<torch::jit::IValue> modelInput;
        std::vector<torch::jit::IValue> lastInput;

    public:
        void setInputParams(int micIndex) {
            memset(&inputParameters, 0, sizeof(inputParameters)); //sets sizeof(inputParameters) number of values at &inputParameters to 0
            inputParameters.channelCount = 1;
            inputParameters.device = micIndex;
            inputParameters.hostApiSpecificStreamInfo = NULL;
            inputParameters.sampleFormat = paInt16;
            inputParameters.suggestedLatency = Pa_GetDeviceInfo(micIndex)->defaultLowInputLatency;
        }

        void openStream() {
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
        }

        void startStream() {
            err = Pa_StartStream(&stream);
            checkErr(err);
            std::cout << "Started stream." << std::endl;
        }

        void stopStream() {
            err = Pa_StopStream(&stream);
            checkErr(err);
            std::cout << "Stopped stream." << std::endl;
        }

        void loadModel() {
            try {
                model = torch::jit::load("/home/dylenthomas/Documents/whisper/.model/whisper.pt");
            } catch (const c10::Error& e) {
                std::cerr << "The model could not be loaded:\n" << e.what();
                exit(0);
            }
            std::cout << "Model loaded successfuly\n";
        }

        int predict() {
            if (modelInput != lastInput){
                {
                    torch::NoGradGuard no_grad; //disable grad
                    torch::Tensor output = model.forward(modelInput).toTensor();
                    return output[0].argmax(-1).item<int>();
                }

                lastInput = modelInput;
            } else {
                return -1;
            }
        }

    private:
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
};

int main() {
    //init portaudio
    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    //find mic index
    const std::string name = "Blue Snowball";
    int micIndex = getMicrophone(name);

    if (micIndex < 0) {
        std::cout << "The device (" << name << ") could not be found.\n";
        std::cout << "This is a list of all the devices:\n";
        printAudioDevices();
        return 0;
    }
    std::cout << "=================================================" << std::endl;
    std::cout << "The device (" << name << ") was found at index: " << micIndex << std::endl;

    Listener listener;
    listener.setInputParams(micIndex);
    listener.loadModel();

    listener.openStream();
    //listener.startStream();




    //shut down
    listener.stopStream();
    err = Pa_Terminate();
    checkErr(err);
    return 0;
}
