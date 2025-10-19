# TensorRT 8.6 to 10.12 Upgrade Changes

This document summarizes the changes made to upgrade the codebase from TensorRT 8.6 to TensorRT 10.12.

## Files Modified

1. `common.hpp`
2. `yolov5_d.cpp`
3. `yolov5_d11.cpp`

## API Changes Applied

### 1. ResizeMode API Change
**Issue**: `ResizeMode::kNEAREST` is not recognized
**Solution**: Replace with `InterpolationMode::kNEAREST`

**Files affected**: `common.hpp`, `yolov5_d.cpp`, `yolov5_d11.cpp`

```cpp
// Before (TensorRT 8.6)
upsample->setResizeMode(ResizeMode::kNEAREST);

// After (TensorRT 10.12)
upsample->setResizeMode(InterpolationMode::kNEAREST);
```

### 2. Builder Configuration Changes
**Issue**: `setMaxBatchSize()` and `setMaxWorkspaceSize()` are deprecated
**Solution**: Remove `setMaxBatchSize()` and replace `setMaxWorkspaceSize()` with `setMemoryPoolLimit()`

**Files affected**: `yolov5_d.cpp`, `yolov5_d11.cpp`

```cpp
// Before (TensorRT 8.6)
builder->setMaxBatchSize(maxBatchSize);
config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB

// After (TensorRT 10.12)
// Note: setMaxBatchSize is deprecated in TensorRT 10.x
config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 16 * (1 << 20));  // 16MB
```

### 3. Memory Management Changes
**Issue**: `destroy()` methods are deprecated
**Solution**: Remove `destroy()` calls as memory is managed automatically

**Files affected**: `yolov5_d.cpp`, `yolov5_d11.cpp`

```cpp
// Before (TensorRT 8.6)
network->destroy();
engine->destroy();
config->destroy();
builder->destroy();
context->destroy();
runtime->destroy();

// After (TensorRT 10.12)
// Note: destroy() is deprecated in TensorRT 10.x, memory is managed automatically
```

### 4. Execution Context Changes
**Issue**: `enqueueV2()` is deprecated, `getBindingIndex()` is deprecated
**Solution**: Replace with `enqueueV3()` and tensor name-based addressing

**Files affected**: `yolov5_d.cpp`, `yolov5_d11.cpp`

```cpp
// Before (TensorRT 8.6)
inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
context.enqueueV2(buffers, stream, nullptr);

// After (TensorRT 10.12)
// Note: In TensorRT 10.x, binding indices are replaced with tensor names
inputIndex = 0;  // Assuming input is at index 0
outputIndex = 1; // Assuming output is at index 1

// In doInference function:
context.setTensorAddress(INPUT_BLOB_NAME, buffers[0]);
context.setTensorAddress(OUTPUT_BLOB_NAME, buffers[1]);
context.enqueueV3(stream);
```

## Compilation Notes

1. Make sure you have TensorRT 10.12 properly installed and linked
2. Update your CMakeLists.txt or build configuration to link against TensorRT 10.12 libraries
3. Verify that all CUDA and cuDNN versions are compatible with TensorRT 10.12
4. Test the inference functionality thoroughly as the execution model has changed

## Potential Issues

1. **Performance**: The new API may have different performance characteristics
2. **Memory Management**: Automatic memory management may behave differently than explicit destroy() calls
3. **Tensor Addressing**: The new tensor name-based addressing may require additional validation
4. **Batch Size**: Without explicit batch size setting, ensure your models handle dynamic batch sizes correctly

## Testing Recommendations

1. Verify that models load correctly
2. Test inference with various input sizes
3. Check memory usage patterns
4. Validate output accuracy against TensorRT 8.6 results
5. Performance benchmarking to ensure no regression