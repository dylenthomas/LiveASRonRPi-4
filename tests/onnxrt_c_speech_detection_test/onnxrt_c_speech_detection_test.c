#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"

int badStatus(const OrtStatus* status, const OrtApi* ort) {
	// Make sure the Api was accessed correctly
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

int kbhit() {
    struct timeval tv = {0L, 0L};
    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(0, &fds);
    return select(1, &fds, NULL, NULL, &tv);
}

int main(int argc, char *argv[]) {
	if (argc != 3) { fprintf("There should be only two args, VAD model path then mic name."); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];

	const int sample_rate = 16000;
	const int buffer_samples = 512;
	const int channels = 1;

	int64_t input_data_shape[] = {1, buffer_samples};
	int64_t state_data_shape[] = {2, 1, 128};

	float buffer[buffer_samples] = {0};
	
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
	if (badStatus(ort->SetGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }

	OrtSession* session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

	OrtMemoryInfo* mem_info = NULL;
	if (badStatus(ort->CreateMemoryInfo("cpu", OrtArenaAllocator, 0, OrtMemTypeCPU, &mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		alloc, sample_rate, sizeof(sample_rate),{1}, 1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, &sr_tensor), ort)) { return 1; }

// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphone...\n");
	snd_pcm_t *mic1_ch;
	
	init_mic(mic1_name, mic1_ch, sample_rate, channels, buffer_samples);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection. Press 'q' to stop.");
	while (1) {
		OrtValue* outputs[2];

		if (kbhit()) {
			char c = getch();
			if (c == 'q') { break; }
		}

		read_mic(buffer, mic1_ch, buffer_size);
		
		if (badStatus(ort->CreateTensorWithDataAsOrtValue(
			alloc, buffer, sizeof(buffer), input_data_shape, 2, 
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }
		
		const OrtValue* const inputs = {input_tensor, state_tensor, sr_tensor};

		if (badStatus(ort-Run(
			session, ort_run_opts, {"input", "state", "sr"}, inputs, 3,
			{"output", "stateN"}, 2, outputs), ort)) { return 1; }

		if (state_tensor) {
			// Make sure to release the previous state tensor once allocated
			if (badStatus(ort->ReleaseValue(state_tensor), ort)) { return 1; }
		}

		// Retrieve probability of speech from the model
		OrtValue* ort_speech_prob = ouputs[0];
		float* prob_data = NULL;
		if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return 1; }
		float speech_prob = prob_data[0];

		fprintf("The probability of speech is: %f\n", speech_prob); 

		// Save the previous state for the next run.
		state_tensor = outputs[1];
		outputs[1] = NULL;

		// release old OrtValues
		if (badStatus(ort->ReleaseValue(input_tensor), ort)) { return 1; }
		if (badStatus(ort->ReleaseValue(outputs[0]), ort)) { return 1; }
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(mem_info);
	ort->ReleaseSession(session);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);
	close_mic(mic2_ch);

	return 0;
}