cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(PORTAUDIO_INCLUDE_DIR "/home/dylenthoms/whisper/include/portaudio/include")
set(PORTAUDIO_LIBRARY "/home/dylenthoms/whisper/include/portaudio/lib/.libs/libportaudio.a")
set(LIB_DIR "/lib/x86_64-linux-gnu")
set(JACK_LIBRARY "${LIB_DIR}/libjack.so.0")
set(JACK_SERVER_LIBRARY "${LIB_DIR}/libjackserver.so.0")
set(JACK_NET_LIBRARY "${LIB_DIR}/libjacknet.so.0")
set(RT_LIBRARY "${LIB_DIR}/librt.so.1")
set(ASOUND_LIBRARY "${LIB_DIR}/libasound.so.2")
#set(CMAKE_PREFIX_PATH "/home/dylenthoms/whisper/include/libtorch")

include_directories(${PORTAUDIO_INCLUDE_DIR})
#find_package(Torch REQUIRED)

add_executable(portaudio_test portaudio_test.cpp)
target_link_libraries(portaudio_test 
#"${TORCH_LIBRARIES}"
"${PORTAUDIO_LIBRARY}"
"${JACK_LIBRARY}"
"${JACK_SERVER_LIBRARY}"
"${JACK_NET_LIBRARY}"
"${RT_LIBRARY}"
"${ASOUND_LIBRARY}"
)
set_property(TARGET portaudio_test PROPERTY CXX_STANDARD 17)