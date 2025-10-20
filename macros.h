#ifndef MACROS_H
#define MACROS_H

// TensorRT API macros for different versions
#ifndef TRT_NOEXCEPT
#define TRT_NOEXCEPT noexcept
#endif

#ifndef TRT_CONST_ENQUEUE
#define TRT_CONST_ENQUEUE const
#endif

// API export/import macros
#ifdef _WIN32
    #ifdef BUILDING_DLL
        #define API __declspec(dllexport)
    #else
        #define API __declspec(dllimport)
    #endif
#else
    #define API
#endif

// CUDA macros
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while(0)
#endif

#endif // MACROS_H