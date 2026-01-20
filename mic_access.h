#ifndef MICACCESS_H_
#define MIC_ACCESS_H_

void read_mic(float* buffer, snd_pcm_t *capture_handle, int buffer_samples);
void init_mic(const char* name, snd_pcm_t *capture_handle, int sample_rate, int channels, int buffer_samples);
void close_mic(snd_pcm_t *capture_handle);

#endif