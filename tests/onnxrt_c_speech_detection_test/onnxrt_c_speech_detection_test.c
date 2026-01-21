#include <stdio.h>
#include <stdlib.h>
#include <sys/select.h>
#include <termios.h>
#include <unistd.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"

int badStatus(OrtStatus* status, const OrtApi* ort) {
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
	if (argc != 3) { printf("There should be only two args, VAD model path then mic name."); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];

	int sample_rate[] = {16000};

	int64_t input_data_shape[] = {1, 512}; // input data will be the mic buffer

	float buffer[512] = {0};

	const char* const input_names_arr = {"input", "state", "sr"};
	const char* const* input_names = &input_names_arr;
	const char* const output_names_arr = {"output", "stateN"};
	const char* const* output_names = &output_names_arr;
	
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

	OrtSession* session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

	OrtMemoryInfo* mem_info = NULL;
	if (badStatus(ort->CreateMemoryInfo("cpu", OrtArenaAllocator, 0, OrtMemTypeCPU, &mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, sample_rate, sizeof(sample_rate), (const int64_t[]){1}, 1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64, &sr_tensor), ort)) { return 1; }

// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphone...\n");
	snd_pcm_t* mic1_ch;
	
	init_mic(mic1_name, &mic1_ch, sample_rate[0], 1, 512);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection. Press any key to stop.");
	while (1) {
		OrtValue* outputs[2];

		if (kbhit()) { break; }

		read_mic(buffer, mic1_ch, 512); // read 512 buffer samples
		
		if (badStatus(ort->CreateTensorWithDataAsOrtValue(
			mem_info, buffer, sizeof(buffer), input_data_shape, 2, 
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }
		
		const OrtValue* const inputs[3] = {input_tensor, state_tensor, sr_tensor};

		if (badStatus(ort->Run(
			session, ort_run_opts, input_names, inputs, 3,
			output_names, 2, outputs), ort)) { return 1; }

		if (state_tensor) {
			// Make sure to release the previous state tensor once allocated
			ort->ReleaseValue(state_tensor);
		}

		// Retrieve probability of speech from the model
		OrtValue* ort_speech_prob = outputs[0];
		float* prob_data = NULL;
		if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return 1; }
		float speech_prob = prob_data[0];

		fprintf(stdout, "The probability of speech is: %f\n", speech_prob); 

		// Save the previous state for the next run.
		state_tensor = outputs[1];
		outputs[1] = NULL;

		// release old OrtValues
		ort->ReleaseValue(input_tensor);
		ort->ReleaseValue(outputs[0]);
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(mem_info);
	ort->ReleaseSession(session);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);

	return 0;
}
