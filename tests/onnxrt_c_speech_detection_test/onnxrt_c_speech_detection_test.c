#include <stdio.h>
#include <conio.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"

int checkOrtStatus(const OrtStatus* status, const OrtApi* ort) {
	// Make sure the Api was accessed correctly
	if (status != NULL) {
		const char* err_msg = ort->GetErrorMessage(status);
		fprintf(stderr, "ORT Error: %s\n", err_msg);
		ort->ReleaseStatus(status);
		return 1;
	}
	return 0;
}

int main(int argc, char *argv[]) {
	if (argc != 3) { fprintf("There should be only two args, VAD model path then mic name."); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];
	
	int intra_threads = 1, inter_threads = 1;

	const int sample_rate = 16000;
	const int buffer_samples = 512;
	const int channels = 1;
	const int state_size = 2 * 1 * 128;

	int64_t input_data_shape[] = {1, buffer_samples};
	int64_t state_data_shape[] = {2, 1, 128};

	float buffer[buffer_samples] = {0};

	const char* const input_node_names = {"input", "state", "sr"};
	const char* const output_node_names = {"output", "stateN"};
// Initialize the ORT Api --------------------------------------------------------------------------
	const OrtApi* ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
	if (ort == NULL) { fprintf(stderr, "ORT api returned nullptr!\n"); return 1; }

	OrtEnv* env = NULL;
	if (checkOrtStatus(ort->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "onnxruntime", &env), ort)) { return 1; }

	OrtSessionOptions* session_opts = NULL;
	if (checkOrtStatus(ort->CreateSessionOptions(&session_opts), ort)) { return 1; }
	if (checkOrtStatus(ort->SetIntraOpNumThreads(session_opts, intra_threads), ort)) { return 1; }
	if (checkOrtStatus(ort->SetInterOpNumThreads(session_opts, inter_threads), ort)) { return 1; }
	if (checkOrtStatus(ort->SetGraphOptimizationLevel(session_opts, ORT_ENABLE_ALL), ort)) { return 1; }

	OrtSession* session = NULL;
	if (checkOrtStatus(ort->CreateSession(env, vad_model_path, session_opts, &session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (checkOrtStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

	OrtMemoryInfo* mem_info = NULL;
	if (checkOrtStatus(ort->CreateMemoryInfo("cpu", OrtArenaAllocator, 0, OrtMemTypeCPU, &mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (checkOrtStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	snd_pcm_t *mic1_ch;
	
	init_mic(mic1_name, mic1_ch, sample_rate, channels, buffer_samples);
// -------------------------------------------------------------------------------------------------
	OrtValue* input_tensor = NULL;
	OrtValue* state_tensor = NULL;
	OrtValue* sr_tensor = NULL;

	if (checkOrtStatus(ort->CreateTensorWithDataAsOrtValue(
		alloc,
		sample_rate,
		sizeof(sample_rate),
		{1},
		1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
		&sr_tensor
		), ort)) { return 1; }

	printf("Starting audio collection. Press 'q' to stop.");
	while (1) {
		if (kbhit()) {
			char c = getch();
			if (c == 'q') { break; }
		}

		read_mic(buffer, mic1_ch, buffer_size);
		if (checkOrtStatus(ort->CreateTensorWithDataAsOrtValue(
			alloc,
			buffer, 
			sizeof(buffer), 
			input_data_shape, 
			2, 
			ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, 
			&input_tensor
			), ort)) { return 1; }
		
		const OrtValue* const inputs = {input_tensor, state_tensor, sr_tensor};
		OrtValue* outputs[2];

		if (checkOrtStatus(ort-Run(
			session,
			ort_run_opts,
			input_node_names,
			inputs,
			3,
			output_node_names,
			2,
			outputs	
			), ort)) { return 1; }

		if (state_tensor) {
			if (checkOrtStatus(ort->ReleaseValue(state_tensor), ort)) { return 1; }
		}

		OrtValue* ort_speech_prob = ouputs[0];
		float* prob_data = NULL;
		if (checkOrtStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { return 1; }
		float speech_prob = prob_data[0];

		fprintf("The probability of speech is: %f\n", speech_prob); 

		state_tensor = outputs[1];
		outputs[1] = NULL;

		// release old OrtValues
		if (checkOrtStatus(ort->ReleaseValue(input_tensor), ort)) { return 1; }
		if (checkOrtStatus(ort->ReleaseValue(outputs[0]), ort)) { return 1; }
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