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
    snd_pcm_hw_params_t *hw_params;

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
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &sample_rate, 0)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror);
        exit(1);
    }

    return 0;
}