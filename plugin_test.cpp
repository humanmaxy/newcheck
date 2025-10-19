#include <iostream>
#include <memory>
#include "yololayer.h"
#include "NvInfer.h"
#include "NvInferPlugin.h"

using namespace nvinfer1;

int main()
{
    std::cout << "Testing YOLO plugin registration..." << std::endl;
    
    try
    {
        // Initialize the plugin library
        bool success = initLibNvInferPlugins(nullptr, "");
        if (!success)
        {
            std::cerr << "Failed to initialize plugin library" << std::endl;
            return -1;
        }
        std::cout << "Plugin library initialized successfully" << std::endl;
        
        // Get the plugin registry
        auto* registry = getPluginRegistry();
        if (!registry)
        {
            std::cerr << "Failed to get plugin registry" << std::endl;
            return -1;
        }
        std::cout << "Plugin registry obtained successfully" << std::endl;
        
        // Try to get the YOLO plugin creator
        auto* creator = registry->getPluginCreator("YoloLayer_TRT", "1");
        if (!creator)
        {
            std::cerr << "Failed to get YOLO plugin creator" << std::endl;
            return -1;
        }
        std::cout << "YOLO plugin creator obtained successfully" << std::endl;
        
        // Test creating a plugin instance
        std::vector<PluginField> fields;
        
        // Create netinfo field
        int netinfo[5] = {17, 640, 640, 1000, 0}; // CLASS_NUM, INPUT_W, INPUT_H, MAX_OUTPUT_BBOX_COUNT, is_segmentation
        PluginField netinfoField{"netinfo", netinfo, PluginFieldType::kINT32, 5};
        fields.push_back(netinfoField);
        
        // Create kernels field
        std::vector<Yolo::YoloKernel> kernels(3);
        kernels[0].width = 80;
        kernels[0].height = 80;
        kernels[1].width = 40;
        kernels[1].height = 40;
        kernels[2].width = 20;
        kernels[2].height = 20;
        
        // Set some example anchors
        float anchors1[] = {10, 13, 16, 30, 33, 23};
        float anchors2[] = {30, 61, 62, 45, 59, 119};
        float anchors3[] = {116, 90, 156, 198, 373, 326};
        
        memcpy(kernels[0].anchors, anchors1, sizeof(anchors1));
        memcpy(kernels[1].anchors, anchors2, sizeof(anchors2));
        memcpy(kernels[2].anchors, anchors3, sizeof(anchors3));
        
        PluginField kernelsField{"kernels", kernels.data(), PluginFieldType::kFLOAT32, kernels.size()};
        fields.push_back(kernelsField);
        
        PluginFieldCollection fc{static_cast<int>(fields.size()), fields.data()};
        
        // Create the plugin
        auto* plugin = creator->createPlugin("yololayer", &fc);
        if (!plugin)
        {
            std::cerr << "Failed to create YOLO plugin instance" << std::endl;
            return -1;
        }
        std::cout << "YOLO plugin instance created successfully" << std::endl;
        
        // Clean up
        plugin->destroy();
        std::cout << "Plugin test completed successfully!" << std::endl;
        
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception during plugin test: " << e.what() << std::endl;
        return -1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception during plugin test" << std::endl;
        return -1;
    }
}