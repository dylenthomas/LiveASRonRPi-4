#include <stdio.h>
#include <stdlib.h>
#include <alsa/asoundlib.h>

/*tutorial ref: https://equalarea.com/paul/alsa-audio.html?utm_source=chatgpt.com*/

int main(int argc, char *argv[]) {
    int i;
    int err;
    unsigned int sample_rate = 44100;
    short buf[128]; /*init an array of shorts named buf that can hold 128 values (which is 256 bytes fun fact)*/
    snd_pcm_t *capture_handle;
    snd_pcm_hw_params_t *hw_params; /* stores all the configurations for the audio device setup */

    /* attempt to reach and open the device */
    if ((err = snd_pcm_open(&capture_handle, argv[1], SND_PCM_STREAM_CAPTURE, 0)) < 0) {
        fprintf(stderr, "cannot open audio device %s (%s)\n", argv[1], snd_strerror(err));
        exit(1);
    }

    /* attempt to allocate hardware parameters */
    if ((err = snd_pcm_hw_params_malloc(&hw_params)) < 0) {
        fprintf(stderr, "cannot allocate hardware parameter structure (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* attempt to intitialize the paramters */
    if ((err = snd_pcm_hw_params_any(capture_handle, hw_params) < 0)) {
        fprintf(stderr, "cannot initialize hardware parameter structure (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* set the access type for the device */
    if ((err = snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED)) < 0) {
        fprintf(stderr, "cannot set access type (%s)\n", snd_strerror);
        exit(1);
    }

    /* set the format for the device */
    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "cannot set sample format (%s)\n", snd_strerror);
        exit(1);
    }

    /* set the sample rate */
    /* this function sends the requested sample rate to the audio device 
       then changes it to the nearest supported sample rate and returns that value
       dir (the last value passed) will be changed based on the rate chosen, and passing 0 means you don't care about the rate chosen */
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &sample_rate, 0)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror);
        exit(1);
    }

    /* attempt to set number of channels */
    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, 2)) < 0) {
        fprintf(stderr, "cannot set channel count (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* apply all the previously set configurations to the sound card */
    if ((err = snd_pcm_hw_params(capture_handle, hw_params)) < 0) {
        fprintf(stderr, "cannot set parameters (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* free the memory used up by the hw_params struct */
    snd_pcm_hw_params_free(hw_params);







    return 0;
}