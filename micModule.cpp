#include <stdio.h>
#include <alsa/asoundlib.h>
#include <cmath>

// g++ -fPIC -shared -o utils/micModule.so micModule.cpp -lasound

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
    short* accessMicrophone(const char* name, unsigned int rate, int channels, int frames, int record_length, int* collected_samples) {
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

        for (int i = 0; i < iters; ++i) {
            if ((err = snd_pcm_readi(capture_handle, buffer, frames)) != frames) {
                fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
                exit(1);
            }

            for (int x = 0; x < frames * channels; ++x) {
                buf_storage[x + (frames * channels * i)] = buffer[x];
            }
        }

        snd_pcm_close(capture_handle);

        *collected_samples = total_samples;
        return buf_storage;
    }
}