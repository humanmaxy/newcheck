#ifndef MACROS_H
#define MACROS_H

#ifdef _WIN32
    #ifdef YOLO_PLUGIN_EXPORTS
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
    
    // Suppress DLL interface warnings for STL containers
    #pragma warning(push)
    #pragma warning(disable: 4251)  // STL containers need DLL interface
    #pragma warning(disable: 4275)  // Non-DLL interface base class
#else
    #define API
#endif

#endif // MACROS_H