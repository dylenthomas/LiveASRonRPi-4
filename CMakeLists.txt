cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(test)

set(PORTAUDIO_INCLUDE_DIR "/home/dylenthomas/Documents/whisper/lib/portaudio/include")
set(PORTAUDIO_LIBRARY "/home/dylenthomas/Documents/whisper/lib/portaudio/lib/.libs/libportaudio.a")
set(LIB_DIR "/lib/x86_64-linux-gnu")
set(JACK_LIBRARY "${LIB_DIR}/libjack.so.0")
set(JACK_SERVER_LIBRARY "${LIB_DIR}/libjackserver.so.0")
set(JACK_NET_LIBRARY "${LIB_DIR}/libjacknet.so.0")
set(RT_LIBRARY "${LIB_DIR}/librt.so.1")
set(ASOUND_LIBRARY "${LIB_DIR}/libasound.so.2")
set(CMAKE_PREFIX_PATH "/home/dylenthomas/Documents/whisper/lib/libtorch")

include_directories(${PORTAUDIO_INCLUDE_DIRS})

find_package(Torch REQUIRED)

add_executable(test listener.cpp)
target_link_libraries(test
    ${TORCH_LIBRARIES}
    ${PORTAUDIO_LIBRARY}
    ${JACK_LIBRARY}
    ${JACK_SERVER_LIBRARY}
    ${JACK_NET_LIBRARY}
    ${RT_LIBRARY}
    ${ASOUND_LIBRARY}
)
set_property(TARGET test PROPERTY CXX_STANDARD 17)
