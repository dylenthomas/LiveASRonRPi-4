#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <cstring>

#include <portaudio.h>
#include <torch/torch.h>

#define SAMPLERATE 44100
#define FRAMESPERBUFFER 44100

using namespace std;

//https://www.youtube.com/watch?v=jpsJCji71Ec

static void checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

int getMicrophone(string name) {
    //Returns the index of the first microphone it finds that contains the name, if no microphone exists then returns -1

    //Init vars
    int numDevices = Pa_GetDeviceCount();
    const PaDeviceInfo* device;
    string deviceName;
    int outputChannels;
    int micIndex = -1;

    //Find the device index
    for (int i=0; i<numDevices; i++) {
        device = Pa_GetDeviceInfo(i);
        deviceName = device->name;
        outputChannels = device->maxOutputChannels;

        //Check if deviceName contains name
        int x = 0;
        while (true) {
            if (deviceName[x] == name[x] && x <= name.length() && outputChannels == 0) {
                x++;
            } else { break; }
        }

        if (x == name.length()) { micIndex = i; break; }
    }

    return micIndex;
}

class listener {
    private:
        PaStreamParameters inputParameters;
        PaError err;
        PaStream* stream;

    public:
        void setInputParams(int micIndex){
            memset(&inputParameters, 0, sizeof(inputParameters));
            inputParameters.channelCount = 1;
            inputParameters.device = micIndex;
            inputParameters.hostApiSpecificStreamInfo = NULL;
            inputParameters.sampleFormat = paInt16;
            inputParameters.suggestedLatency = Pa_GetDeviceInfo(micIndex)->defaultLowInputLatency;
        }

        void openStream(){
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
        }

        void startStream(){
            err = Pa_StartStream(&stream);
            checkErr(err);
        }

        void stopStream(){
            err = Pa_StopStream(&stream);
            checkErr(err);
        }


    private:
        static int callback(
                const void* inputBuffer,
                void* outputBuffer,
                unsigned long framesPerBuffer,
                const PaStreamCallbackTimeInfo* timeInfo,
                PaStreamCallbackFlags statusFlags,
                void* userData
                ){
            int* input = (int*)inputBuffer;
            (void)outputBuffer; //prevent unused var error

            return 0;
        }
};

int main() {
    //Start up
    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    //Find the microphone index
    const string name = "iPhone";
    int micIndex = getMicrophone(name);

    if (micIndex < 0) {
        printf("The device: %s was not found.\n", name.c_str());
        return EXIT_FAILURE;
    } else {
        printf("Device (%s) found at index: %d\n", name.c_str(), micIndex);
    }










    //Shut down
    err = Pa_Terminate();
    checkErr(err);

    return EXIT_SUCCESS;
}
