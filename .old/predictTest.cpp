#include <alsa/asoundlib.h>

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#define SAMPLE_RATE 16000;
#define CHANNELS 1;
#define FRAMES 256;
#define SECONDS_TO_RECORD 5;

torch::jit::script:Module model;
std::vector<torch::jit::IValue> input_tensor;
torch::Tensor output_tensor;

void checkErr(int err, int check_value) {
    if (err < check_value) {
        fprintf(stderr, "error! %s", snd_strerror(err));
        exit(1);
    }
}

void predict(std::vector<torch::jit:IValue> input, torch::Tensor* output) {
    {
        torch::NoGradGuard no_grad;
        model.eval();
        *ouput = model.forward(input).toTensor();
    }
}

struct decoder : torch::nn::Module {
    torch::Tensor forward(torch::)
}

int main(int argc, char *argv[]) {
    int err;
    unsigned int rate = SAMPLE_RATE;
    int num_samples = SAMPLE_RATE * SECONDS_TO_RECORD;
    int iters = std::round(num_samples / FRAMES);

    short buffer[FRAMES];
    short storage[FRAMES * iters];

    snd_pcm_t *capture_handle;
    snd_pcm_hw_params_t *hw_params;

    int runs = 0;

    if (argc != 3) {
        std::cerr << "usage predictTest <path/to/model> <audio device name>" << std::endl;
        exit(1);
    }

    try {
        model = torch::jit::load(argv[1]);
    }
    catch(const c10::Error& e) {
        std::cerr << "error loading model\n";
        exit(1);
    }

    checkErr(snd_pcm_open(&capture_handle, argv[1], SND_PCM_STREAM_CAPTURE, 0)), 0);
    checkErr(snd_pcm_hw_params_malloc(&hw_params), 0);
    checkErr(snd_pcm_hw_params_any(capture_handle, hw_params), 0);
    checkErr(snd_pcm_hw_params_set_access(capture_handle, hw_params, SND_PCM_ACCESS_RW_INTERLEAVED), 0);
    checkErr(snd_pcm_hw_params_set_format(capture_handle, hw_params, SND_PCM_FORMAT_S16_LE), 0);
    checkErr(snd_pcm_hw_params_set_rate_near(capture_handle, hw_params, &rate, 0), 0);
    checkErr(snd_pcm_hw_params_set_channels(capture_handle, hw_params, CHANNELS), 0);
    checkErr(snd_pcm_hw_params(capture_handle, hw_params), 0);
    snd_pcm_hw_params_free(hw_params);
    checkErr(snd_pcm_prepare(capture_handle), 0);

    for (int i = 0; i < iters, ++i) {
        if (snd_pcm_readi(capture_handle, buffer, FRAMES) != FRAMES) {
            fprintf(stderr, "read from audio interface failed (%s)\n", snd_strerror(err));
            exit(1);
        }

        /* coppy the buffer into long_term storage */
        for (int x = 0; x < FRAMES; ++x) {
            storage[x + (FRAMES * runs)] = buffer[x];
        }
        runs++;
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    input_tensor.push_back(torch::from_blob(storage, {1, FRAMES * iters}));

    predict(input_tensor, &output_tensor);






    snd_pcm_close(capture_handle);

    return 0;
}