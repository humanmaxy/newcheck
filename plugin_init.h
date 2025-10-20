#ifndef PLUGIN_INIT_H
#define PLUGIN_INIT_H

#include <NvInfer.h>

// Function to initialize TensorRT plugins
// This should be called before creating any TensorRT networks
extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace);

// Helper function to ensure plugins are initialized
inline bool initializeTensorRTPlugins() {
    static bool initialized = false;
    if (!initialized) {
        initialized = initLibNvInferPlugins(nullptr, "");
    }
    return initialized;
}

#endif // PLUGIN_INIT_H