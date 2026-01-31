#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

#include "onnxruntime_c_api.h"
#include "mic_access.h"
#include "keywordConfParser.h"
#include "transcripter_api.h"

#define KEYWORD_CONF "configs/keywords.json"
#define MIC_BUFFER_LEN 512 
#define SAMPLE_RATE 16000

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

void getSpeechProb(
        float* speech_prob,
		const OrtApi* ort,
		OrtSession* session,
		OrtRunOptions* run_ops,
		OrtValue* input_tensor,
		OrtValue* state_tensor,
		OrtValue* sr_tensor
		) {
	static const char* input_names[] = {"input", "state", "sr"};
	static const char* output_names[] = {"output", "stateN"};

	const OrtValue* const inputs[3] = {input_tensor, state_tensor, sr_tensor};
	OrtValue* outputs[2] = {NULL, NULL};
	
	if (badStatus(ort->Run(
		session, run_opts, input_names, inputs, 3,
		output_names, 2, outputs), ort)) 
    { 
        speech_prob = NULL; 
        return;
    }

	// Retrieve probability of speech from the model
	OrtValue* ort_speech_prob = outputs[0];
	
	float* prob_data = NULL;
	if (badStatus(ort->GetTensorMutableData(ort_speech_prob, (void**)&prob_data), ort)) { 
        speech_prob = NULL;
        return; 
    }
	*speech_prob = prob_data[0];

	// release old OrtValues
	ort->ReleaseValue(state_tensor);
	ort->ReleaseValue(outputs[0]);
	
	// Save the previous state for the next run.
	state_tensor = outputs[1];
	outputs[1] = NULL;
}

// TODO implement auto read onnx model parameters
int main(int argc, char *argv[]) {
    Py_Initialize();
    PyInit_transcripter();

	if (argc != 3) { printf("There should be only two args, VAD model path then mic name.\n"); return 0; }
	const char* vad_model_path = argv[1];
	const char* mic1_name = argv[2];

    const struct keywordHM keywords;
    keywords = createKeywordHM(KEYWORD_CONF);

	const float speech_threshold = 0.7; // trigger threshold to start transcription

	const int64_t sample_rate[] = {SAMPLE_RATE};
	const int64_t sample_rate_shape[] = {1};

	const int64_t input_data_shape[] = {1, MIC_BUFFER_LEN}; // input data will be the mic buffer
	const int64_t state_data_shape[] = {2, 1, 128};	

	int16_t tmp_buffer[MIC_BUFFER_LEN] = {0};
    float long_buffer[SAMPLE_RATE * 30] = {0}; // 30 seconds is the most audio transcript model can take
	float buffer[MIC_BUFFER_LEN] = {0};
	float initial_state[2 * 1 * 128] = {0};

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

	OrtSession* ort_session = NULL;
	if (badStatus(ort->CreateSession(env, vad_model_path, session_opts, &ort_session), ort)) { return 1; }

	OrtRunOptions* ort_run_opts = NULL;
	if (badStatus(ort->CreateRunOptions(&ort_run_opts), ort)) { return 1; }

	OrtMemoryInfo* mem_info = NULL;
	if (badStatus(ort->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &mem_info), ort)) { return 1; }

	OrtAllocator* alloc = NULL;
	if (badStatus(ort->GetAllocatorWithDefaultOptions(&alloc), ort)) { return 1; }
	
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, sample_rate, sizeof(sample_rate), sample_rate_shape, 1,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &sr_tensor), ort)) { return 1; }

	// create initializing state for model of all zeros 
	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, initial_state, sizeof(initial_state), state_data_shape, 3,
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &state_tensor), ort)) { return 1; }

	if (badStatus(ort->CreateTensorWithDataAsOrtValue(
		mem_info, buffer, sizeof(buffer), input_data_shape, 2, 
		ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor), ort)) { return 1; }
// -------------------------------------------------------------------------------------------------
// Initialize Microphone ---------------------------------------------------------------------------
	printf("Initializing microphone...\n");
	snd_pcm_t* mic1_ch;
	init_mic(mic1_name, &mic1_ch, sample_rate[0], 1, MIC_BUFFER_LEN);
// -------------------------------------------------------------------------------------------------
	printf("Starting audio collection.\n");
	while (1) {
		float speech_prob;
	
		int i = 0;
		read_mic(tmp_buffer, mic1_ch, MIC_BUFFER_LEN);
        // convert int mic data to float	
		while (i < MIC_BUFFER_LEN) { buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; }
		
		getSpeechProb(&speech_prob, ort, ort_session, ort_run_ops, input_tensor, state_tensor, sr_tensor);
        if (speech_prob == NULL) { continue; } // if VAD failed just skip the iteration 
        else if (*speech_prob < speech_threshold) { continue; } // if not enough skip the iteration
        
        int z = 0;
        while (*speech_prob >= speech_threshold) {
			// collect audio data for transcription
            i = 0;
		    read_mic(tmp_buffer, mic1_ch, MIC_BUFFER_LEN);
            while (i < MIC_BUFFER_LEN) { buffer[i] = (float)tmp_buffer[i] / 32768.0f; i++; } // convert int mic data to float
			
            getSpeechProb(&speech_prob, ort, ort_session, ort_run_ops, input_tensor, state_tensor, sr_tensor);
            if (speech_prob == NULL) { break; } // if VAD failed stop there
           
            i = 0;
            while(i < MIC_BUFFER_LEN) { long_buffer[i + MIC_BUFFER_LEN * z] = buffer[i]; i++;}
            z++;
		}
        // should keep track of z because z = the number of buffers read, so we know where to 
        // stop transcribing in the long buffer
	}
	
// Cleanup for program exit ------------------------------------------------------------------------
	ort->ReleaseMemoryInfo(mem_info);
	ort->ReleaseSession(ort_session);
	ort->ReleaseSessionOptions(session_opts);
	ort->ReleaseEnv(env);

	close_mic(mic1_ch);

    Py_Finalize();

	return 0;
}
