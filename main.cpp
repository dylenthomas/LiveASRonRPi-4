#include <alsa/asoundlib.h>

/* Tutorial referrence: https://alsamodular.sourceforge.net/alsa_programming_howto.html */

int main() {
    /* Handle for the PCM device */
    snd_pcm_t *pcm_handle;

    /* Playback stream */
    snd_pcm_stream_t stream = SND_PCM_STREAM_PLAYBACK;

    /* This structure contains information about the hardware that can be used to
       specify the configuration to be used for the PCM stream 
    */
    snd_pcm_hw_params_t *hwparams;

    /* Name of the PCM device, like plughw:0, 0 
       the first number is the number of the soundcard,
       the second number is the number of the device. 
    */
    char *pcm_name = strdup("plughw:0,0");

    /* Allocate the snd_pcm_hw_params_t structure on the stack */
    snd_pcm_hw_params_alloca(&hwparams);

    /* Open PCM. The last parameter of this function is the mode. 
       If this is set to 0, the standard mode is used.
       If SND_PCM_NONBLOCK is used, read/write access to the PCM device will return immediately.
       If SND_PCM_ASYNC is specified, SIGIO will be emitted whenever a period has been completely processed by the soundcard.
    */
    if (snd_pcm_open(&pcm_handle, pcm_name, stream, 0) < 0) {
        fprintf(stderr, "Error opening PCM device: %s\n", pcm_name);
        return -1;
    }

    /* Iit hwparams with full configuration space */
    if (snd_pcm_hw_params_any(pcm_handle, hwparams) < 0) {
        fprintf(stderr, "Can not configure this PCM device.\n");
        return -1;
    }

    int rate = 44100;
    unsigned int exact_rate; /* Sample rate returned by snd_pcm_params_set_rate_near */
    int dir; 
    /* exact_rate == rate --> dir = 0 
       exact_rate < rate --> dir = -1
       exact_rate > rate --> dir = 1
    */
    int periods = 2;
    snd_pcm_uframes_t periodsize = 8192; /* periodsize in bytes */

    /* Set access type.
       This can be either SND_PCM_ACCESS_RW_INTERLEAVED or SND_PCM_ACCESS_RW_NONINTERLEAVED. 
    */
    if (snd_pcm_hw_params_set_access(pcm_handle, hwparams, SND_PCM_ACCESS_RW_INTERLEAVED) < 0) {
        fprintf(stderr, "Error setting access.\n");
        return -1;
    } 

    /* Set sample format */
    if(snd_pcm_hw_params_set_format(pcm_handle, hwparams, SND_PCM_FORMAT_S16_LE) < 0) {
        fprintf(stderr, "Error setting format.\n");
        return -1;
    }

    /* Set sample rate, if exact rate is not supported by the hardware, use closest rate */
    exact_rate = rate;
    if (snd_pcm_hw_params_set_rate_near(pcm_handle, hwparams, &exact_rate, 0) < 0) {
        fprintf(stderr, "Error setting rate.\n");
        return -1;
    }
    if (rate != exact_rate) {
        fprintf(stderr, "The rate %d Hz is not supported by your hardware.\n==>Using %d Hz instead.\n", rate, exact_rate);
    }

    /* Set number of channels */
    if (snd_pcm_hw_params_set_channels(pcm_handle, hwparams, 2) < 0) {
        fprintf(stderr, "Error setting channels.\n");
        return -1;
    }

    /* Set number of periods */
    if (snd_pcm_hw_params_set_periods(pcm_handle, hwparams, periods, 0) < 0) {
        fprintf(stderr, "Error setting periods.\n");
        return -1;
    }

    /* Set buffer size (in frames). The resulting latency is given by: latency = periodsize * periods / (rate * bytes_per_frame) */
    if (snd_pcm_hw_params_set_buffer_size(pcm_handle, hwparams, (periodsize * periods) >> 2) < 0) {
        fprintf(stderr, "Error setting buffersize.\n");
        return -1;
    }

    /* Apply HW parameter settings to PCM device and prepare device */
    if (snd_pcm_hw_params(pcm_handle, hwparams) < 0) {
        fprintf(stderr, "Error setting HW params.\n");
        return -1;
    }

    /* Write num_frames frames from buffer data to the PCM device pointed to by pcm_handle. 
       Returns the number of frames actually written.
    */
    snd_pcm_sframes_t snd_pcm_writei(pcm_handle, data, num_frames);

    return 0;
}