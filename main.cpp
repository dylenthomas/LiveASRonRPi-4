#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <alsa/asoundlib.h>

#define SAMPLE_RATE 44100
#define CHANNELS 1
#define FRAMES 256
#define ITERATIONS 1000
#define FILE_NAME "test.wav"

#define WRITE_BUFFER(file, buf, count) fwrite(buf, sizeof(buf[0]), count, file)

/*tutorial ref: https://equalarea.com/paul/alsa-audio.html?utm_source=chatgpt.com*/

void write_wav_header(FILE *file, int sample_rate, int num_samples) {
    int bytes_per_sample = 2;

    int byte_rate = sample_rate * CHANNELS * bytes_per_sample; /* calculate bytes per second */
    int data_size = num_samples * CHANNELS * bytes_per_sample; /* calculate how many bytes will be stored */
    int chunk_size = 36 + data_size; /* the size of the file started at byte 8 (since the first 8 aren't counted) */

    /* write the RIFF format header */
    fwrite("RIFF", 1, 4, file); /* array "RIFF" is 4 things 1 bytes each */
    fwrite(&chunk_size, 4, 1, file); /* int chunk_size is 1 thing 4 bytes each */
    fwrite("WAVE", 1, 4, file); /* same as "RIFF" */

    /* write the fmt subchunk */
    fwrite("fmt ", 1, 4, file);

    uint32_t subchunk1_size = 16; /* this is always 16 and must be 4 bytes large */
    /* set the format as 16-bit PCM */
    uint16_t audio_format = 1; /* this specifies the audio device as a PCM, and must be 2 bytes large */
    uint16_t bits_per_sample = 16; /* this specifies there are 2 bytes per sample, or 16 bits, and is a part of the audio format itself [16-but PCM] */
    /* redefine channels type */
    uint16_t channels = CHANNELS;
    uint16_t block_align = channels * 2;

    /* start wrinting subchunk info to the fmt subchunk */
    fwrite(&subchunk1_size, 4, 1, file);
    fwrite(&audio_format, 2, 1, file);
    fwrite(&channels, 2, 1, file);
    fwrite(&sample_rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file);
    fwrite(&block_align, 2, 1, file);
    fwrite(&bits_per_sample, 2, 1, file);

    /* write the data subchunk */
    fwrite("data", 1, 4, file);
    fwrite(&data_size, 4, 1, file);
}


int main(int argc, char *argv[]) {
    int i;
    int err;
    unsigned int rate = SAMPLE_RATE;
    short buf[FRAMES]; /*init an array of shorts named buf that can hold FRAMES values (which is FRAMES * 2 bytes fun fact)*/
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
        fprintf(stderr, "cannot set access type (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* set the format for the device */
    if ((err = snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE)) < 0) {
        fprintf(stderr, "cannot set sample format (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* set the sample rate */
    /* this function sends the requested sample rate to the audio device 
       then changes it to the nearest supported sample rate and returns that value
       dir [the last value passed] will be changed based on the rate chosen, and passing 0 means you don't care about the rate chosen */
    if ((err = snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0)) < 0) {
        fprintf(stderr, "cannot set sample rate (%s)\n", snd_strerror(err));
        exit(1);
    }
    printf("actual rate: (%u)\n", rate);
    
    /* attempt to set number of channels */
    if ((err = snd_pcm_hw_params_set_channels(capture_handle, hw_params, CHANNELS)) < 0) {
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

    if ((err = snd_pcm_prepare(capture_handle)) < 0) {
        fprintf(stderr, "cannot prepare audio interface (%s)\n", snd_strerror(err));
        exit(1);
    }

    /* read from the audio device and save the data */
    int num_samples = FRAMES * ITERATIONS;
    FILE *outfile = fopen(FILE_NAME, "wb");
    write_wav_header(outfile, SAMPLE_RATE, num_samples);

    printf("starting audio capture...\n");

    for (i = 0; i < ITERATIONS; ++i) {
        if ((err = snd_pcm_readi(capture_handle, buf, FRAMES)) != FRAMES) {
            fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
            exit(1);
        }

        /* write the buffer to the file */
        WRITE_BUFFER(outfile, buf, err * CHANNELS); /* on success err is the number of frames read */
    }

    /* finish with the file and audio device */
    fclose(outfile);
    snd_pcm_close(capture_handle);
    printf("saved %d samples to %s\n", num_samples, FILE_NAME);

    return 0;
}