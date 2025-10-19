#ifndef PLUGIN_INIT_H
#define PLUGIN_INIT_H

// Forward declaration - must be at global scope
extern "C" bool initLibNvInferPlugins(void* logger, const char* libNamespace);

// Safe plugin initialization function
// Call this before using TensorRT engines that require the YoloLayer plugin
inline bool initializeYoloPlugin() {
    // This function is implemented in yololayer_impl.cpp
    return initLibNvInferPlugins(nullptr, "");
}

#endif // PLUGIN_INIT_H