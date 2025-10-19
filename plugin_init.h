#ifndef PLUGIN_INIT_H
#define PLUGIN_INIT_H

#include <NvInfer.h>

// Safe plugin initialization function
// Call this before using TensorRT engines that require the YoloLayer plugin
inline bool initializeYoloPlugin() {
    // This function is implemented in yololayer.cpp
    extern bool initLibNvInferPlugins(void* logger, const char* libNamespace);
    return initLibNvInferPlugins(nullptr, "");
}

#endif // PLUGIN_INIT_H