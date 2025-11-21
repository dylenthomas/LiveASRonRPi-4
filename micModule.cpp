#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <stdio.h>
#include <cmath>
#include <iostream>
#include <vector>
#include <sstream>
#include <cstring>
#include <limits>
#include <chrono>
#include <iomanip>
#include <memory>
#include <string>
#include <stdexcept>
#include <cstdio>
#include <cstdarg>
#if __cplusplus < 201703L
#include <memory>
#endif

#include <alsa/asoundlib.h>
#include "onnxruntime_cxx_api.h"

// Build command:
// g++ -fPIC -shared -o utils/micModule.so micModule.cpp -I /home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/include/ -L /home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/lib  -lasound -lonnxruntime -Wl,-rpath,/home/dylenthomas/networkShare/mnemosyne/calliope/LARS/onnxruntime-linux-x64-1.12.1/lib

class VAD {
private:
    // ONNX Runtime resources
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::shared_ptr<Ort::SessionOptions> session = nullptr;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCPU(OrtArenaAllocator, OrtMemTypeCPU);

    // context
    const int context_samples = 64; // for 16kHz use 64 samples as context
    std::vector<float> _context;

    int window_size_samples;
    int effective_window_size; // window_size_samples + context_samples

    int samples_per_ms;

    // ONNX Runtime input/output buffers
    std::vector<Ort::Value> ort_inputs;
    std::vector<const char*> input_node_names = {"input", "state", "sr"};
    std::vector<float> input;
    unsigned int size_state = 2 * 1 * 128;
    std::vector<float> _state;
    std::vector<int64_t> sr;
    int64_t input_node_dims[2] = {};
    const int64_t state_node_dims[3] = {2, 1, 128};
    const int64_t sr_node_dims[1] = {1};
    std::vector<Ort::Value> ort_outputs;
    std::vector<const char*> output_node_names = {"output", "stateN"};

    // Model params
    int sample_rate;
    float threshold;
    int min_silence_samples;
    int min_silence_samples_at_max_speech;
    int min_speech_samples;
    float max_speech_samples;
    int speech_pad_samples;
    int audio_length_samples;

    // State management
    bool triggered = false;
    unsigned int temp_end = 0;
    unsigned int current_sample = 0;
    int prev_end;
    int next_start = 0;

    void init_onnx_model(const std::wstring& model_path) {
        init_engine_threads(1, 1);
        session = std::make_shared<Ort::Session>(env, model_path.c_str(), session_options);
    }

    void init_engine_threads(int inter_threads, int intra_threads) {
        session_options.SetIntraOpNumThreads(intra_threads);
        session_options.SetInterOpNumThreads(inter_threads);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLEvel::ORT_ENABLE_ALL);
    }

    void reset_states() {
        std::memset(_state.data(), 0, _state.size() * sizeof(float))
        triggered = false;
        temp_end = 0;
        current_sample = 0;
        prev_end = next_start = 0;
        std::fill(_context.begin(), _context.end(), 0.0f);
    }

    // run inference on a single mic buffer
    void predict(const std::vector<float>& buffer) {
        // build a new input: [context, buffer]
        std::vector<float> new_buffer(effective_window_size, 0.0f);
        std::copy(_context.begin(), _context.end(), new_buffer.begin());
        std::copy(buffer.begin(), buffer.end(), new_buffer.begin() + context_samples);
        input = new_buffer;

        Ort::Value input_ort = Ort::Value::CreateTensor<float>(
            memory_info, input.data(), input.size(), input_node_dims, 2);
        Ort::Value state_ort = Ort::Value::CreateTensor<float>(
            memory_info, _state.data(), _state.size(), state_node_dims, 3);
        Ort::Value sr_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, sr.data(), sr.size(), sr_node_dims, 1);
        ort_inputs.clear();
        ort_inputs.emplace_back(std::move(input_ort));
        ort_inputs.emplace_back(std::move(state_ort));
        ort_inputs.emplace_back(std::move(sr_ort));

        ort_outputs = session ->Run(
            Ort::RunOptions{ nullptr },
            input_node_names.data(), ort_inputs.data(), ort_inputs.size(),
            output_node_names.data(), output_node_names.size()
        );

        float speech_prob = ort_outputs[0].GetTensorMutableData<float>()[0];
        float* stateN = ort_outputs[1].GetTensorMutableData<float>();
        std::memcpy(_state.data(), stateN, size_state * sizeof(float));
    }

public: // Constructor
    VAD(const std::wstring model_path, int rate,) {
        this->rate = sample_rate;

        init_onnx_model(model_path)
    }
}

    // https://github.com/snakers4/silero-vad/blob/master/examples/cpp/silero-vad-onnx.cpp#L4

    // figure out if I can create a main function in here that initializes the VAD model, so I don't have to worry about creating a function to do so, and its jsut done on import. 

extern "C" {
    // function to reset VAD states 
}

void checkErr(int err, int check_val) {
    if (err < check_val) {
        fprintf(stderr, "error!, %s\n", snd_strerror(err));
        exit(1);
    }
}

extern "C" {
    /* free the returned audio data buffer in python from memory since python doesn't own that memory */
    void freeBuffer(short* buffer) {
        delete[] buffer;
    }
}

extern "C" {
    // Change this so it returns a variable length buffer with a max length so it only returns large buffers with speech in them, which simplifies the Python code. 
    
    short* accessMicrophone(const char* name, unsigned int rate, int channels, int frames, float record_length, int* collected_samples) {
        int iters = std::round((rate * record_length) / frames);
        int total_samples = frames * channels * iters;
        int err;
        
        short buffer[frames * channels];
        /* allocate array on the heap, so they can be transferred to python since they survive after the function ends 
           returning something like short buf_storage[total_smaples] would return a pointer ot dead memory */
        short* buf_storage = new short[total_samples];

        snd_pcm_t *capture_handle;
        snd_pcm_hw_params_t *hw_params;
        
        checkErr(snd_pcm_open(&capture_handle, name, SND_PCM_STREAM_CAPTURE, 0), 0);
        checkErr(snd_pcm_hw_params_malloc(&hw_params), 0);
        checkErr(snd_pcm_hw_params_any(capture_handle, hw_params), 0);
        checkErr(snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED), 0);
        checkErr(snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE), 0);
        checkErr(snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0), 0);
        checkErr(snd_pcm_hw_params_set_channels(capture_handle, hw_params, channels), 0);
        checkErr(snd_pcm_hw_params(capture_handle, hw_params), 0);
        snd_pcm_hw_params_free(hw_params);
        checkErr(snd_pcm_prepare(capture_handle), 0);

        for (int i = 0; i < iters; i++) {
            if ((err = snd_pcm_readi(capture_handle, buffer, frames)) != frames) {
                fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
                exit(1);
            }

            // high pass filter

            // if any of the buffers have speech over the threshold return the buffer 

            for (int x = 0; x < frames * channels; x++) { // place buffer in long term storage
                buf_storage[x + (frames * channels * i)] = buffer[x];
            }
        }

        snd_pcm_close(capture_handle);

        *collected_samples = total_samples;
        return buf_storage;
    }
}