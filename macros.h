#ifndef MACROS_H
#define MACROS_H

#include <NvInfer.h>

// Define API macro for Windows DLL export/import
#ifdef _WIN32
    #ifdef PLUGIN_EXPORTS
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
    // For plugin creators, we need special handling
    #ifdef BUILDING_PLUGIN_LIB
        #define PLUGIN_API __declspec(dllexport)
    #else
        #define PLUGIN_API
    #endif
#else
    #define API
    #define PLUGIN_API
#endif

// Safe plugin registration macro that avoids access violations
#define REGISTER_TENSORRT_PLUGIN(name) \
    extern "C" { \
        bool initLibNvInferPlugins(void* logger, const char* libNamespace) { \
            static name pluginCreator; \
            getPluginRegistry()->registerCreator(pluginCreator, libNamespace ? libNamespace : ""); \
            return true; \
        } \
    }

#endif // MACROS_H