#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "config_parser.h"
#include "transcripter_api.h"

#define PROGRAM_CONF "configs/program.json"
#define KEYWORD_CONF "configs/keywords.json"
#define MIC_BUFFER_LEN 512 
#define SAMPLE_RATE 16000
#define CHANNELS 1
#define LONGEST_TRANSCRIPT 30

#define STATE_LEN 2 * 1 * 128
#define SAMPLE_RATE_DIMS 1
#define INPUT_DIMS 2
#define STATE DIMS 3

#define VENDOR_ID 4318
#define DEVICE_ID 0
#define GPU_ALIGNMENT 0

int badStatus(OrtStatus* status, const OrtApi* ort) {
	// Make sure the API was accessed correctly 
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

void getSpeechProb(
        float* speech_prob,
        OrtValue** outputs,
        size_t output_count,
		const OrtApi* ort,
		OrtSession* session,
		OrtRunOptions* run_opts,
        OrtIoBinding* io_binding,
        OrtAllocator* alloc
		) {
    if (badStatus(ort->RunWithBinding(session, run_opts, io_binding), ort)) { return; }
    if (badStatus(ort->GetBoundOutputValues(io_binding, alloc, &outputs, &output_count), ort)) { return; }
	
	// Retrieve probability of speech from the model
	OrtValue* ort_speech_prob = outputs[0];
	
	float* prob_data = NULL;
	if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return; }
	*speech_prob = prob_data[0];
}

// TODO implement auto read onnx model parameters
int main(int argc, char *argv[]) {
    Py_Initialize();
    PyInit_transcripter();

	if (argc != 4) { printf("Args should be: VAD Model Path, Mic1 Name, Mic2 Name\n"); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];
    const char* mic2_name = argv[3];

    const struct keywordHM keywords;
    keywords = createKeywordHM(KEYWORD_CONF);

	const float speech_threshold = 0.7; // trigger threshold to start transcription
    size_t num_outputs = 2;
    
    double peak_value = 0.0f;
    int hold_iterations = 5;
    int iterations_held = 0;
    double decay_rate = 0.25f;
    int iterations_decayed = 0;

    int transcript_buffers = 0;
	
    const int64_t sample_rate[] = {SAMPLE_RATE};
	const int64_t sample_rate_shape[] = {1};

	const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
	const int64_t state_data_shape[] = {2, 1, 128};	

	int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
    float long_buffer[SAMPLE_RATE * LONGEST_TRANSCRIPT] = {0};
	float buffer[MIC_BUFFER_LEN] = {0};
	float initial_state[STATE_LEN] = {0};

	OrtValue* input_tensor = NULL;
	OrtValue* state_tensor = NULL;
	OrtValue* sr_tensor = NULL;
// Initialize the ORT Api --------------------------------------------------------------------------
	printf("Initializing ORT...\n");
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return 1; }

	OrtEnv* env = NULL;
	if (badStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &env), ort)) { return 1; }

	OrtSessionOptions* session_opts = NULL;
	if (badStatus(ort->CreateSessionOptions(&session_opts), ort)) { return 1; }
	if (badStatus(ort->SetIntraOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetInterOpNumThreads(session_opts, 1), ort)) { return 1; }
	if (badStatus(ort->SetSessionGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }

    OrtCUDAProviderOptionsV2* cuda_opts = NULL;
    if (badStatus(ort->CreateCUDAProviderOptions(&cuda_opts), ort)) { return 1; }
    if (badStatus(ort->SessionOptionsAppendExecutionProvider_CUDA_V2(session_opts, cuda_opts), ort)) { return 1; }

	OrtSession* ort_session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &ort_session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

    OrtMemoryInfo* gpu_mem_info = NULL;
    if (badStatus(ort->CreateMemoryInfo_V2("Cuda", OrtMemoryInfoDeviceType_GPU, VENDOR_ID, DEVICE_ID,
            OrtMemTypeDefault, GPU_ALIGNMENT, OrtDeviceAllocator, &gpu_mem_info), ort)) { return 1; }

	OrtMemoryInfo* cpu_mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &cpu_mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, SAMPLE_RATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { return 1; }

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, initial_state, sizeof(initial_state), state_data_shape, STATE_DIMS,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { return 1; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		gpu_mem_info, buffer, sizeof(buffer), input_data_shape, INPUT_DIMS, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }

    OrtIoBinding* io_binding = NULL;
    if (badStatus(ort->CreateIoBinding(session, &io_binding), ort)) { return 1; }

    if (badStatus(ort->BindInput(io_binding, "input", input_tensor), ort)) { return 1; }
    if (badStatus(ort->BindInput(io_binding, "state", state_tensor), ort)) { return 1; }
    if (badStatus(ort->BindInput(io_binding, "sr", sr_tensor), ort)) { return 1; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "output", cpu_mem_info), ort)) { return 1; }
    if (badStatus(ort->BindOutputToDevice(io_binding, "stateN", cpu_mem_info), ort)) { return 1; }
// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphones...\n");

	snd_pcm_t* mic1_ch;
	init_mic(mic1_name, &mic1_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);

    snd_pcm_t* mic2_ch;
    init_mic(mic2_name, &mic2_ch, SAMPLE_RATE, CHANNELS, MIC_BUFFER_LEN);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection.\n");
    
    while (1) {
		float* speech_prob = NULL;
        OrtValue** outputs = NULL;
	
		int i = 0;
		read_mic(tmp_buffer, mic1_ch, MIC_BUFFER_LEN);
        // convert int mic data to float	
		while (i < MIC_BUFFER_LEN) { buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
		
		getSpeechProb(speech_prob, outputs, num_outputs, ort, ort_session, ort_run_ops, io_binding, alloc);
        if (speech_prob == NULL) { continue; } // if VAD failed just skip the iteration 

        if (*speech_prob > peak_value) {
        // Increase
            peak_value = *speech_prob;
            iterations_decayed = 0;
        }
        else if (iterations_held <= hold_iterations) {
        // Hold
            iterations_held++;
        }
        else { 
        // Decay
            iterations_held = 0;
            peak_value *= exp(-1 * decay_rate * iterations_decayed);
            iterations_decayed++;
        }

        if (peak_value >= speech_threshold) {
            i = 0;
            while (i < MIC_BUFFER_LEN) { long_buffer[i + MIC_BUFFER_LEN * transcript_buffers] = buffer[i]; i++; }
            transcript_buffers++;
        }
        else if (transcript_buffers) {
            // transcribe

            long_buffer = {0};
            transcript_buffers = 0;
        }

        // Cleanup iteration
        ort->ReleaseValue(state_tensor);
        ort->ReleaseValue(outputs[0]);
        state_tensor = outputs[1];
        outputs[1] = NULL;
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(cpu_mem_info);
    ort->ReleaseMemoryInfo(gpu_mem_info);
    ort->ReleaseIoBinding(io_binding);
	ort->ReleaseSession(ort_session);
    ort->ReleaseCUDAProviderOptions(cuda_opts);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);
    close_mic(mic2_ch);

    Py_Finalize();

	return 0;
}
