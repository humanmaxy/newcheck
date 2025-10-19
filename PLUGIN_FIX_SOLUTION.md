# TensorRT 10.x 插件注册访问冲突修复方案

## 问题描述

错误发生在 `PluginRegistrar` 构造函数中，具体位置：
```
nvinfer_10.dll!00007ffeb671c238() - 访问冲突: 读取位置 0x0000000000000001
```

这是由于 TensorRT 10.x 版本升级后，插件注册机制发生变化导致的。

## 根本原因

1. **插件注册时机问题**: 全局静态初始化时 `getPluginRegistry()` 可能返回无效指针
2. **TensorRT 版本兼容性**: TensorRT 10.x 的插件 API 发生了变化
3. **缺少 yololayer.h 文件**: 项目中引用了 `yololayer.h` 但文件可能不存在或不完整

## 解决方案

### 方案 1: 创建兼容的 yololayer.h 文件

需要创建一个与 TensorRT 10.x 兼容的 `yololayer.h` 文件：

```cpp
#ifndef YOLO_LAYER_H
#define YOLO_LAYER_H

#include <vector>
#include <string>
#include <NvInfer.h>

namespace Yolo {
    static constexpr int CHECK_COUNT = 3;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 17;  // 根据你的类别数调整
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;

    struct Detection {
        float bbox[4];  // x, y, w, h
        float conf;     // confidence
        float class_id; // class id
    };

    struct YoloKernel {
        int width;
        int height;
        float anchors[CHECK_COUNT * 2];
    };
}

// TensorRT 10.x 兼容的插件创建器
class YoloPluginCreator : public nvinfer1::IPluginCreator {
public:
    YoloPluginCreator();
    ~YoloPluginCreator() override = default;

    const char* getPluginName() const noexcept override;
    const char* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const char* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

// 安全的插件注册函数
void registerYoloPlugin();

#endif // YOLO_LAYER_H
```

### 方案 2: 修改初始化流程

在 `Detection_J::Initialize` 函数开始时调用插件注册：

```cpp
bool Detection_J::Initialize(int camType, const char* model_path, const char* config_path, int num) {
    // 确保插件已注册
    static bool pluginRegistered = false;
    if (!pluginRegistered) {
        try {
            registerYoloPlugin();
            pluginRegistered = true;
        } catch (const std::exception& e) {
            std::cerr << "插件注册失败: " << e.what() << std::endl;
            return false;
        }
    }
    
    // 继续原有的初始化流程...
}
```

### 方案 3: 使用预编译的引擎文件

如果插件问题难以解决，可以：

1. 使用已经编译好的 `.engine` 文件，跳过网络构建过程
2. 确保 `.engine` 文件是用相同版本的 TensorRT 生成的

```cpp
// 修改模型加载逻辑，直接加载 .engine 文件
std::string engine_path = std::string(model_path);
if (engine_path.find(".engine") == std::string::npos) {
    std::cerr << "错误: 请使用 .engine 格式的模型文件" << std::endl;
    return false;
}
```

## 实施步骤

1. **立即修复**: 使用方案 3，确保使用正确的 `.engine` 文件
2. **短期修复**: 实施方案 2，添加安全的插件注册
3. **长期修复**: 实施方案 1，创建完整的 TensorRT 10.x 兼容插件

## 验证步骤

1. 确认所有 `.engine` 文件存在且可访问
2. 验证 TensorRT 版本与 CUDA 版本兼容
3. 测试模型加载和推理功能
4. 检查内存使用情况

## 注意事项

- 确保所有依赖库版本兼容
- 在生产环境部署前充分测试
- 保留原始的 TensorRT 8.6 版本作为备份