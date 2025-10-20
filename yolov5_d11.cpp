#include "pch.h"
#include"yololayer.h"
#include"yolov5_d.h"
#include "plugin_init.h" 
#include <corecrt_io.h>
#include <mutex>
#include <tchar.h> // 解决debug下的报错
//static const int INPUT_H = Yolo::INPUT_H;
//static const int INPUT_W = Yolo::INPUT_W;
//static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
//const char* INPUT_BLOB_NAME = "images";
//const char* OUTPUT_BLOB_NAME = "output0";
//static Logger gLogger;

using namespace std;
const char* INPUT_BLOB_NAME = "images";
const char* OUTPUT_BLOB_NAME = "output0";
static Logger gLogger;

int piyi = 68;

// 读取配置ini文件
bool Detection_J::ReadParams(const std::string& file_path)
{
	/*int config_data;*/
	std::string config_path = file_path;

	if (_access(config_path.c_str(), 0) == -1)
		return false;

	LPTSTR path = new TCHAR[config_path.size() + 1];
	//strcpy(path, config_path.c_str());
#ifdef UNICODE
	size_t size = config_path.size() + 1;
	mbstowcs(path, config_path.c_str(), size);
#else
	// 如果使用 ANSI
	strcpy(path, config_path.c_str());
#endif
	//calibration = ::GetPrivateProfileInt("base", "calibration", 0.062, path);
	//loujinshu = ::GetPrivateProfileInt("base", "loujinshu", 0.5, path);
	//area_loujinshu = ::GetPrivateProfileInt("base", "loujinshu_area", 6, path);
	calibration = ::GetPrivateProfileInt(_T("base"), _T("calibration"), 0.062, path);
	loujinshu = ::GetPrivateProfileInt(_T("base"), _T("loujinshu"), 0.5, path);
	area_loujinshu = ::GetPrivateProfileInt(_T("base"), _T("loujinshu_area"), 6, path);
	return true;
}

static int get_width(int x, float gw, int divisor = 8) {
	return int(ceil((x * gw) / divisor)) * divisor;
}

static int get_depth(int x, float gd) {
	if (x == 1) return 1;
	int r = round(x * gd);
	if (x * gd - int(x * gd) == 0.5 && (int(x * gd) % 2) == 0) {
		--r;
	}
	return std::max<int>(r, 1);
}

ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
	INetworkDefinition* network = builder->createNetworkV2(0U);

	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);
	std::map<std::string, Weights> weightMap = loadWeights(wts_name);
	/* ------ yolov5 backbone------ */
	auto conv0 = convBlock(network, weightMap, *data, get_width(64, gw), 6, 2, 1, "model.0");
	assert(conv0);
	auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	auto bottleneck_CSP2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	auto conv3 = convBlock(network, weightMap, *bottleneck_CSP2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	auto bottleneck_csp4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
	auto conv5 = convBlock(network, weightMap, *bottleneck_csp4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	auto bottleneck_csp6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	auto conv7 = convBlock(network, weightMap, *bottleneck_csp6->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.7");
	auto bottleneck_csp8 = C3(network, weightMap, *conv7->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
	auto spp9 = SPPF(network, weightMap, *bottleneck_csp8->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.9");
	/* ------ yolov5 head ------ */
	auto conv10 = convBlock(network, weightMap, *spp9->getOutput(0), get_width(512, gw), 1, 1, 1, "model.10");

	auto upsample11 = network->addResize(*conv10->getOutput(0));
	assert(upsample11);
	upsample11->setResizeMode(ResizeMode::kNEAREST);
	upsample11->setOutputDimensions(bottleneck_csp6->getOutput(0)->getDimensions());

	ITensor* inputTensors12[] = { upsample11->getOutput(0), bottleneck_csp6->getOutput(0) };
	auto cat12 = network->addConcatenation(inputTensors12, 2);
	auto bottleneck_csp13 = C3(network, weightMap, *cat12->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.13");
	auto conv14 = convBlock(network, weightMap, *bottleneck_csp13->getOutput(0), get_width(256, gw), 1, 1, 1, "model.14");

	auto upsample15 = network->addResize(*conv14->getOutput(0));
	assert(upsample15);
	upsample15->setResizeMode(ResizeMode::kNEAREST);
	upsample15->setOutputDimensions(bottleneck_csp4->getOutput(0)->getDimensions());

	ITensor* inputTensors16[] = { upsample15->getOutput(0), bottleneck_csp4->getOutput(0) };
	auto cat16 = network->addConcatenation(inputTensors16, 2);

	auto bottleneck_csp17 = C3(network, weightMap, *cat16->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.17");

	/* ------ detect ------ */
	IConvolutionLayer* det0 = network->addConvolutionNd(*bottleneck_csp17->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.0.weight"], weightMap["model.24.m.0.bias"]);
	auto conv18 = convBlock(network, weightMap, *bottleneck_csp17->getOutput(0), get_width(256, gw), 3, 2, 1, "model.18");
	ITensor* inputTensors19[] = { conv18->getOutput(0), conv14->getOutput(0) };
	auto cat19 = network->addConcatenation(inputTensors19, 2);
	auto bottleneck_csp20 = C3(network, weightMap, *cat19->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.20");
	IConvolutionLayer* det1 = network->addConvolutionNd(*bottleneck_csp20->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.1.weight"], weightMap["model.24.m.1.bias"]);
	auto conv21 = convBlock(network, weightMap, *bottleneck_csp20->getOutput(0), get_width(512, gw), 3, 2, 1, "model.21");
	ITensor* inputTensors22[] = { conv21->getOutput(0), conv10->getOutput(0) };
	auto cat22 = network->addConcatenation(inputTensors22, 2);
	auto bottleneck_csp23 = C3(network, weightMap, *cat22->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.23");
	IConvolutionLayer* det2 = network->addConvolutionNd(*bottleneck_csp23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.24.m.2.weight"], weightMap["model.24.m.2.bias"]);

	auto yolo = addYoLoLayer(network, weightMap, "model.24", std::vector<IConvolutionLayer*>{det0, det1, det2});
	yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*yolo->getOutput(0));
	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
	config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
	std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
	assert(builder->platformHasFastInt8());
	config->setFlag(BuilderFlag::kINT8);
	Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
	config->setInt8Calibrator(calibrator);
#endif

	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;

	// Don't need the network any more
	network->destroy();

	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}

	return engine;
}

ICudaEngine* build_engine_p6(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name) {
	INetworkDefinition* network = builder->createNetworkV2(0U);
	// Create input tensor of shape {3, INPUT_H, INPUT_W} with name INPUT_BLOB_NAME
	ITensor* data = network->addInput(INPUT_BLOB_NAME, dt, Dims3{ 3, INPUT_H, INPUT_W });
	assert(data);

	std::map<std::string, Weights> weightMap = loadWeights(wts_name);

	/* ------ yolov5 backbone------ */
	auto conv0 = convBlock(network, weightMap, *data, get_width(64, gw), 6, 2, 1, "model.0");
	auto conv1 = convBlock(network, weightMap, *conv0->getOutput(0), get_width(128, gw), 3, 2, 1, "model.1");
	auto c3_2 = C3(network, weightMap, *conv1->getOutput(0), get_width(128, gw), get_width(128, gw), get_depth(3, gd), true, 1, 0.5, "model.2");
	auto conv3 = convBlock(network, weightMap, *c3_2->getOutput(0), get_width(256, gw), 3, 2, 1, "model.3");
	auto c3_4 = C3(network, weightMap, *conv3->getOutput(0), get_width(256, gw), get_width(256, gw), get_depth(6, gd), true, 1, 0.5, "model.4");
	auto conv5 = convBlock(network, weightMap, *c3_4->getOutput(0), get_width(512, gw), 3, 2, 1, "model.5");
	auto c3_6 = C3(network, weightMap, *conv5->getOutput(0), get_width(512, gw), get_width(512, gw), get_depth(9, gd), true, 1, 0.5, "model.6");
	auto conv7 = convBlock(network, weightMap, *c3_6->getOutput(0), get_width(768, gw), 3, 2, 1, "model.7");
	auto c3_8 = C3(network, weightMap, *conv7->getOutput(0), get_width(768, gw), get_width(768, gw), get_depth(3, gd), true, 1, 0.5, "model.8");
	auto conv9 = convBlock(network, weightMap, *c3_8->getOutput(0), get_width(1024, gw), 3, 2, 1, "model.9");
	auto c3_10 = C3(network, weightMap, *conv9->getOutput(0), get_width(1024, gw), get_width(1024, gw), get_depth(3, gd), true, 1, 0.5, "model.10");
	auto sppf11 = SPPF(network, weightMap, *c3_10->getOutput(0), get_width(1024, gw), get_width(1024, gw), 5, "model.11");

	/* ------ yolov5 head ------ */
	auto conv12 = convBlock(network, weightMap, *sppf11->getOutput(0), get_width(768, gw), 1, 1, 1, "model.12");
	auto upsample13 = network->addResize(*conv12->getOutput(0));
	assert(upsample13);
	upsample13->setResizeMode(ResizeMode::kNEAREST);
	upsample13->setOutputDimensions(c3_8->getOutput(0)->getDimensions());
	ITensor* inputTensors14[] = { upsample13->getOutput(0), c3_8->getOutput(0) };
	auto cat14 = network->addConcatenation(inputTensors14, 2);
	auto c3_15 = C3(network, weightMap, *cat14->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.15");

	auto conv16 = convBlock(network, weightMap, *c3_15->getOutput(0), get_width(512, gw), 1, 1, 1, "model.16");
	auto upsample17 = network->addResize(*conv16->getOutput(0));
	assert(upsample17);
	upsample17->setResizeMode(ResizeMode::kNEAREST);
	upsample17->setOutputDimensions(c3_6->getOutput(0)->getDimensions());
	ITensor* inputTensors18[] = { upsample17->getOutput(0), c3_6->getOutput(0) };
	auto cat18 = network->addConcatenation(inputTensors18, 2);
	auto c3_19 = C3(network, weightMap, *cat18->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.19");

	auto conv20 = convBlock(network, weightMap, *c3_19->getOutput(0), get_width(256, gw), 1, 1, 1, "model.20");
	auto upsample21 = network->addResize(*conv20->getOutput(0));
	assert(upsample21);
	upsample21->setResizeMode(ResizeMode::kNEAREST);
	upsample21->setOutputDimensions(c3_4->getOutput(0)->getDimensions());
	ITensor* inputTensors21[] = { upsample21->getOutput(0), c3_4->getOutput(0) };
	auto cat22 = network->addConcatenation(inputTensors21, 2);
	auto c3_23 = C3(network, weightMap, *cat22->getOutput(0), get_width(512, gw), get_width(256, gw), get_depth(3, gd), false, 1, 0.5, "model.23");

	auto conv24 = convBlock(network, weightMap, *c3_23->getOutput(0), get_width(256, gw), 3, 2, 1, "model.24");
	ITensor* inputTensors25[] = { conv24->getOutput(0), conv20->getOutput(0) };
	auto cat25 = network->addConcatenation(inputTensors25, 2);
	auto c3_26 = C3(network, weightMap, *cat25->getOutput(0), get_width(1024, gw), get_width(512, gw), get_depth(3, gd), false, 1, 0.5, "model.26");

	auto conv27 = convBlock(network, weightMap, *c3_26->getOutput(0), get_width(512, gw), 3, 2, 1, "model.27");
	ITensor* inputTensors28[] = { conv27->getOutput(0), conv16->getOutput(0) };
	auto cat28 = network->addConcatenation(inputTensors28, 2);
	auto c3_29 = C3(network, weightMap, *cat28->getOutput(0), get_width(1536, gw), get_width(768, gw), get_depth(3, gd), false, 1, 0.5, "model.29");

	auto conv30 = convBlock(network, weightMap, *c3_29->getOutput(0), get_width(768, gw), 3, 2, 1, "model.30");
	ITensor* inputTensors31[] = { conv30->getOutput(0), conv12->getOutput(0) };
	auto cat31 = network->addConcatenation(inputTensors31, 2);
	auto c3_32 = C3(network, weightMap, *cat31->getOutput(0), get_width(2048, gw), get_width(1024, gw), get_depth(3, gd), false, 1, 0.5, "model.32");

	/* ------ detect ------ */
	IConvolutionLayer* det0 = network->addConvolutionNd(*c3_23->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.0.weight"], weightMap["model.33.m.0.bias"]);
	IConvolutionLayer* det1 = network->addConvolutionNd(*c3_26->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.1.weight"], weightMap["model.33.m.1.bias"]);
	IConvolutionLayer* det2 = network->addConvolutionNd(*c3_29->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.2.weight"], weightMap["model.33.m.2.bias"]);
	IConvolutionLayer* det3 = network->addConvolutionNd(*c3_32->getOutput(0), 3 * (Yolo::CLASS_NUM + 5), DimsHW{ 1, 1 }, weightMap["model.33.m.3.weight"], weightMap["model.33.m.3.bias"]);

	auto yolo = addYoLoLayer(network, weightMap, "model.33", std::vector<IConvolutionLayer*>{det0, det1, det2, det3});
	yolo->getOutput(0)->setName(OUTPUT_BLOB_NAME);
	network->markOutput(*yolo->getOutput(0));

	// Build engine
	builder->setMaxBatchSize(maxBatchSize);
	config->setMaxWorkspaceSize(16 * (1 << 20));  // 16MB
#if defined(USE_FP16)
	config->setFlag(BuilderFlag::kFP16);
#elif defined(USE_INT8)
	std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
	assert(builder->platformHasFastInt8());
	config->setFlag(BuilderFlag::kINT8);
	Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, INPUT_W, INPUT_H, "./coco_calib/", "int8calib.table", INPUT_BLOB_NAME);
	config->setInt8Calibrator(calibrator);
#endif

	std::cout << "Building engine, please wait for a while..." << std::endl;
	ICudaEngine* engine = builder->buildEngineWithConfig(*network, *config);
	std::cout << "Build engine successfully!" << std::endl;

	// Don't need the network any more
	network->destroy();

	// Release host memory
	for (auto& mem : weightMap)
	{
		free((void*)(mem.second.values));
	}

	return engine;
}

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, bool& is_p6, float& gd, float& gw, std::string& wts_name) {
	// Create builder
	IBuilder* builder = createInferBuilder(gLogger);
	IBuilderConfig* config = builder->createBuilderConfig();

	// Create model to populate the network, then set the outputs and create an engine
	ICudaEngine* engine = nullptr;
	if (is_p6) {
		engine = build_engine_p6(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
	}
	else {
		engine = build_engine(maxBatchSize, builder, config, DataType::kFLOAT, gd, gw, wts_name);
	}
	assert(engine != nullptr);

	// Serialize the engine
	(*modelStream) = engine->serialize();

	// Close everything down
	engine->destroy();
	config->destroy();
	builder->destroy();

}

void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize) {
	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
	CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueueV2(buffers, stream, nullptr);
	CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);
}

bool parse_args(int argc, char** argv, std::string& wts, std::string& engine, bool& is_p6, float& gd, float& gw, std::string& img_dir) {
	if (argc < 4) return false;
	if (std::string(argv[1]) == "-s" && (argc == 5 || argc == 7)) {
		wts = std::string(argv[2]);
		engine = std::string(argv[3]);
		auto net = std::string(argv[4]);
		if (net[0] == 'n') {
			gd = 0.33;
			gw = 0.25;
		}
		else if (net[0] == 's') {
			gd = 0.33;
			gw = 0.50;
		}
		else if (net[0] == 'm') {
			gd = 0.67;
			gw = 0.75;
		}
		else if (net[0] == 'l') {
			gd = 1.0;
			gw = 1.0;
		}
		else if (net[0] == 'x') {
			gd = 1.33;
			gw = 1.25;
		}
		else if (net[0] == 'c' && argc == 7) {
			gd = atof(argv[5]);
			gw = atof(argv[6]);
		}
		else {
			return false;
		}
		if (net.size() == 2 && net[1] == '6') {
			is_p6 = true;
		}
	}
	else if (std::string(argv[1]) == "-d" && argc == 4) {
		engine = std::string(argv[2]);
		img_dir = std::string(argv[3]);
	}
	else {
		return false;
	}
	return true;
}

//static int get_width(int x, float gw, int divisor = 8) {
//	//return math.ceil(x / divisor) * divisor
//	if (int(x * gw) % divisor == 0) {
//		return int(x * gw);
//	}
//	return (int(x * gw / divisor) + 1) * divisor;
//}
//
//static int get_depth(int x, float gd) {
//	if (x == 1) {
//		return 1;
//	}
//	else {
//		return round(x * gd) > 1 ? round(x * gd) : 1;
//	}
//}

int ImContrastAdjust(const cv::Mat& input_img, int min_input_val, int max_input_val, int min_output_val, int max_output_val, cv::Mat& stretch_img)
{
	if (input_img.empty())
		return -1;

	if (input_img.type())
		return -2;

	if (min_input_val < 0)
		min_input_val = 0;

	if (max_input_val > 255)
		max_input_val = 255;

	if (min_input_val > max_input_val)
	{
		int temp = min_input_val;
		min_input_val = max_input_val;
		max_input_val = temp;
	}
	try
	{
		uchar lut_data[256];
		for (int i = 0; i < 256; ++i)
		{
			if (i <= min_input_val)
			{
				lut_data[i] = min_output_val;
			}

			if (i > min_input_val && i < max_input_val)
			{
				lut_data[i] = 1.0 * (max_output_val - min_output_val) / (max_input_val - min_input_val) * (i - min_input_val) + min_output_val;
			}

			if (i >= max_input_val)
			{
				lut_data[i] = max_output_val;
			}
		}
		cv::Mat lut_mat(1, 256, CV_8UC1, lut_data);
		cv::LUT(input_img, lut_mat, stretch_img);
		return 0;
	}
	catch (const cv::Exception& e)
	{
		spdlog::get("CATL_WCP")->info(std::string(e.what()));
		//LogWriterFlush(e.what());
		return -3;
	}
	catch (const std::exception& e)
	{
		return -4;
	}
}

void calculate_contrast(const cv::Mat& defect_img, int thresh, int& defect_area, int& contrast ,int mean_val)
{
	if (defect_img.empty())
	{
		spdlog::get("CATL_WCP")->info("Defect image is empty.");
		//LogWriterFlush("Defect image is empty.");
		contrast = 0;
		return;
	}
	try
	{
		cv::Mat mask;
		if (defect_img.channels() == 3)
			cv::cvtColor(defect_img, mask, cv::COLOR_BGR2GRAY);
		else
			mask = defect_img;
		ImContrastAdjust(mask, mean_val, 100, 0, 255, mask);
		cv::threshold(mask, mask, thresh, 255, cv::THRESH_BINARY);
		defect_area = 0;

		// 前景像素均值
		int defect_mean_val = cv::mean(defect_img, mask)[0];
		int background_mean_val = cv::mean(defect_img, ~mask)[0];
		contrast = abs(defect_mean_val - background_mean_val);
	}
	catch (const cv::Exception& e)
	{
		// 出现异常，当作缺陷拦截
		contrast = 100;
		spdlog::get("CATL_WCP")->info(std::string(e.what()));
		//LogWriterFlush(e.what());
	}
}


//void doInference(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* input, float* output, int batchSize) {
//	// DMA input batch data to device, infer on the batch asynchronously, and DMA output back to host
//	CUDA_CHECK(cudaMemcpyAsync(buffers[0], input, batchSize * 3 * INPUT_H * INPUT_W * sizeof(float), cudaMemcpyHostToDevice, stream));
//	//context.enqueue(batchSize, buffers, stream, nullptr);
//	context.enqueueV2(buffers, stream, nullptr);
//	CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchSize * OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
//	cudaStreamSynchronize(stream);
//}

bool Detection_J::Initialize(int cameraType, const char* model_path, const char* config_path,  int num)
{
	camType = cameraType;
	if (num < 0 || num>3) {
		std::cout << "=================="
			"0, yolov5s"
			"1, yolov5m"
			"2, yolov5l"
			"3, yolov5x" << std::endl;
	}
	std::cout << "Net use :" << yolo_nets[num].netname << std::endl;
	this->gd = yolo_nets[num].gd;
	this->gw = yolo_nets[num].gw;

	//初始化GPU引擎
	cudaSetDevice(DEVICE);
	std::ifstream file(model_path, std::ios::binary);
	if (!file.good()) {
		spdlog::get("CATL_WCP")->info("read mode file error!");
		//LogWriterFlush("read mode file error!");
		return false;
	}
	//if (!ReadParams(config_path))
	//{
	//	LogWriterFlush("base config failed.");
	//	return false;
	//}

	//初始化检测参数
	std::ifstream in(config_path);
	std::string line;
	bool beginPT1 = false;
	bool beginPT2 = false;


	if (in)
	{
		while (getline(in, line))
		{
			if (line == "[/ParamsThresh]")
				break;

			if (line == "</ParamsThresh>")
				break;

			if (beginPT1)
			{
				std::string::size_type pos = line.find("=");
				std::string key = line.substr(0, pos);
				std::string val = line.substr(pos + 1);
				m_defect_thresh_map[key] = std::stof(val);
			}
			if (line == "[ParamsThresh]")
				beginPT1 = true;

			if (beginPT2)
			{
				std::string::size_type pos = line.find("=");
				std::string key = line.substr(0, pos);
				std::string val = line.substr(pos + 1);
				m_defect_thresh_map[key] = std::stof(val);
			}
			if (line == "<ParamsThresh>")
				beginPT2 = true;
		}
	}
	else
	{
		spdlog::get("CATL_WCP")->info("defect config failed.");
		//LogWriterFlush("defect config failed.");
		return false;
	}
	file.seekg(0, file.end);
	size = file.tellg();                // 统计模型字节流大小
	file.seekg(0, file.beg);
	trtModelStream = new char[size];    // 申请模型字节流大小的空间
	assert(trtModelStream);
	file.read(trtModelStream, size);    // 读取字节流到trtModelStream
	file.close();


	// prepare input data ------NCHW---------------------
	runtime = createInferRuntime(gLogger);
	assert(runtime != nullptr);
	engine = runtime->deserializeCudaEngine(trtModelStream, size);
	assert(engine != nullptr);
	context = engine->createExecutionContext();
	assert(context != nullptr);
	delete[] trtModelStream;
	assert(engine->getNbBindings() == 2);
	inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
	outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
	assert(inputIndex == 0);
	assert(outputIndex == 1);
	// Create GPU buffers on device
	CUDA_CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
	CUDA_CHECK(cudaStreamCreate(&stream));
	spdlog::get("CATL_WCP")->info("model Initialize successfully!");
	//LogWriterFlush("model Initialize successfully!");
	return true;
}

bool Detection_J::reload(const char* config_path)
{
	std::ifstream in(config_path);
	std::string line;
	bool beginPT = false;
	spdlog::get("CATL_WCP")->info("begin to reload params");
	//LogWriterFlush("begin to reload params.");
	std::map<std::string, float> temp_map;
	if (in)
	{
		while (getline(in, line))
		{
			if (line == "[/ParamsThresh]")
				break;
			if (beginPT)
			{
				std::string::size_type pos = line.find("=");
				std::string key = line.substr(0, pos);
				std::string val = line.substr(pos + 1);
				temp_map[key] = std::stof(val);
			}
			if (line == "[ParamsThresh]")
				beginPT = true;
		}
	}
	else
	{
		spdlog::get("CATL_WCP")->info("reload config failed");
		//LogWriterFlush("reload config failed.");
		return false;
	}
	
	auto iter1 = m_defect_thresh_map.begin();
	if (m_defect_thresh_map.begin()->second == 0)
	{
		++iter1;
	}
	auto iter2 = temp_map.begin();
	for (; iter1 != m_defect_thresh_map.end(); ++iter1,++iter2)
	{
		if (iter1->second != iter2->second)
		{
			spdlog::get("CATL_WCP")->info("参数变更：" + iter1->first + "由" + std::to_string(iter1->second)
				+ "变更为" + std::to_string(iter2->second));
			/*LogWriterFlush("参数变更：" + iter1->first + "由" + std::to_string(iter1->second)
			+ "变更为" + std::to_string(iter2->second));*/
		}
	}
	m_defect_thresh_map = temp_map;
	return true;
}

static void color_transfer_with_spilt(cv::Mat& input, std::vector<cv::Mat>& chls)
{
	cv::cvtColor(input, input, cv::COLOR_BGR2YCrCb);
	cv::split(input, chls);
}

static void color_retransfer_with_merge(cv::Mat& output, std::vector<cv::Mat>& chls)
{
	cv::merge(chls, output);
	cv::cvtColor(output, output, cv::COLOR_YCrCb2BGR);
}

cv::Mat clahe_deal(cv::Mat& src)
{
	cv::Mat ycrcb = src.clone();
	std::vector<cv::Mat> channels;

	color_transfer_with_spilt(ycrcb, channels);

	cv::Mat clahe_img;
	cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
	// 直方图的柱子高度大于计算后的ClipLimit的部分被裁剪掉，然后将其平均分配给整张直方图
	// 从而提升整个图像
	clahe->setClipLimit(15.);    // (int)(4.*(8*8)/256)
	clahe->setTilesGridSize(cv::Size(4, 4)); // 将图像分为8*8块
	clahe->apply(channels[0], clahe_img);
	channels[0].release();
	clahe_img.copyTo(channels[0]);
	color_retransfer_with_merge(ycrcb, channels);
	return ycrcb;

}

void DetectBubble(cv::Mat& srcImg, std::vector<DefectData>& defect_data, int img_x, int img_w, int y1, int y2, std::map<std::string, float>& m_defect_thresh_map) {
	int qp_width_min = 0;
	int qp_height_min = 0;
	float dazhou_disthre = 0.5 / 0.078;
	int area_qipao_min = 2400;
	dazhou_disthre = m_defect_thresh_map["dazhou_disthre"] / 0.078;
	qp_width_min = m_defect_thresh_map["qp_width_min"] / 0.078;
	qp_height_min = m_defect_thresh_map["qp_height_min"] / 0.078;
	area_qipao_min = m_defect_thresh_map["area_qipao_min"] / 0.078 /0.078;
	float dazhou_dis = 0;
	try
	{
		//cv::Mat temp;
		std::vector<cv::Mat> BGR;

		//int tj_x, tj_y, tj_w, tj_h;
		//tj_x = r_tiejiao.x * res_w;
		//tj_y = r_tiejiao.y * res_h;
		//tj_h = r_tiejiao.height * res_h;
		//tj_w = r_tiejiao.width * res_w;
		//cv::Rect tiejiaoRt(tj_x, tj_y, tj_w, tj_h);
		//cv::Mat temp = srcImg(tiejiaoRt);

		//cv::Rect tiejiaoRt(r_tiejiao.x, r_tiejiao.y, r_tiejiao.width, r_tiejiao.height);
		cv::Rect tiejiaoRt(0, y1, img_w - 1, y2 - y1);
		cv::Mat tiejiao_crop = srcImg(tiejiaoRt).clone();

		cv::split(tiejiao_crop, BGR);

		////**2024/10/25lilu**抓到胶纸打皱与气泡需要检测的区域
		cv::Mat det_bin;
		cv::threshold(BGR[2], det_bin, 25, 255, cv::THRESH_BINARY_INV);
		cv::Mat kernel_jiao = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10));
		cv::morphologyEx(det_bin, det_bin, cv::MORPH_CLOSE, kernel_jiao);
		std::vector<std::vector<cv::Point>> contours1;
		cv::findContours(det_bin, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

		cv::Mat det_region = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
		for (int i = 0; i < contours1.size(); i++) {
			double area = cv::contourArea(contours1[i]);
			//if (area > areaMin) {
			if (area > 100000) {
				cv::RotatedRect t1 = cv::minAreaRect(contours1[i]);
				cv::Point2f rect[4];
				t1.points(rect);
				std::vector<cv::Point> trOutCroppedInt;
				for (int j = 0; j < 4; j++)
				{
					trOutCroppedInt.emplace_back(cv::Point2i(rect[j].x, rect[j].y));
				}
				cv::fillConvexPoly(det_region, trOutCroppedInt, 255);

				//cv::drawContours(det_region, contours1, static_cast<int>(i), cv::Scalar(255), -1, 1);
			}
		}


		//拿到缺陷区域
		cv::threshold(BGR[2], tiejiao_crop, 35, 255, cv::THRESH_BINARY);
		cv::Mat kerne55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
		cv::morphologyEx(tiejiao_crop, tiejiao_crop, cv::MORPH_OPEN, kerne55);
		cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15));
		cv::morphologyEx(tiejiao_crop, tiejiao_crop, cv::MORPH_CLOSE, kernel);
		

		//把属于检测区域的缺陷抠出来
		cv::Mat rest;
		cv::copyTo(tiejiao_crop, rest, det_region);
		cv::Mat kernel5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 35));
		cv::morphologyEx(rest, rest, cv::MORPH_OPEN, kernel5);


		cv::Mat labels, centroids, stats, res_img;
		int connected_num = 0;
		std::vector<std::vector<cv::Point>> contours;
		cv::findContours(rest, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
		cv::Mat dst = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
		double maxArea = 0;
		int maxIdx = 0;
		//int areaMin = 10000;
		int /*areaMin = 2400,*/ areaMax = 500000;
		for (int i = 0; i < contours.size(); i++) {
			double area = cv::contourArea(contours[i]);
			//if (area > areaMin) {
			if (area > area_qipao_min && area < areaMax) {

				maxArea = area;
				maxIdx = i;
				cv::drawContours(dst, contours, static_cast<int>(maxIdx), cv::Scalar(255), -1, 1);
			}
		}

		/*int maxAreaLabel = 0;
		maxArea = 0;
		int x, y, w, h;
		int x_min = 9999;*/



		cv::Mat grayTemp;
		cv::threshold(tiejiao_crop, grayTemp, 200, 255, cv::THRESH_BINARY_INV);
		//cv::morphologyEx(grayTemp, grayTemp, cv::MORPH_OPEN, kernel);
		//maxAreaLabel = 0;
		//try
		//{
		//	connected_num = cv::connectedComponentsWithStats(grayTemp, labels, stats, centroids);
		//}
		//catch (const std::exception&)
		//{
		//	spdlog::get("CATL_WCP")->error("无连通域");
		//	//LogWriterFlush("无连通域");
		//	return;
		//}


		//for (int i = 1; i < connected_num; i++)
		//{
		//	int x_temp = stats.at<int>(i, 0);
		//	if (x_temp < x_min) {
		//		x_min = x_temp;
		//		maxAreaLabel = i;
		//	}
		//	x = stats.at<int>(maxAreaLabel, 0);
		//	y = stats.at<int>(maxAreaLabel, 1);
		//	w = stats.at<int>(maxAreaLabel, 2);
		//	h = stats.at<int>(maxAreaLabel, 3);
		//}

		/*for (int i = 1; i < connected_num; i++)
		{
			int x_temp = stats.at<int>(i, 0);
			if (x_temp < x_min) {
				x_min = x_temp;
				maxAreaLabel = i;
			}
			x = stats.at<int>(maxAreaLabel, 0);
			y = stats.at<int>(maxAreaLabel, 1);
			w = stats.at<int>(maxAreaLabel, 2);
			h = stats.at<int>(maxAreaLabel, 3);
		}*/

		/*cv::Mat add_out;
		cv::add(BGR[2], dst, add_out);
		cv::Rect detectRt;
		int detectRt_x = x + w < 0 ? 0 : x + w;
		int detectRt_y = 0;
		int detectRt_w = add_out.cols - x - w - 10 > 0 ? add_out.cols - x - w - 10 : 0;
		int detectRt_h = add_out.rows;
		detectRt = cv::Rect(detectRt_x, detectRt_y, detectRt_w, detectRt_h);*/
		
		//cv::Mat detectImg = add_out(detectRt);
		//cv::Mat detectGray;

		////cv::inRange(detectImg, 30, 254, detectGray);
		//cv::inRange(detectImg, 240, 255, detectGray);

		//// guoyg05 20241102
		//cv::Mat det_region_crop = det_region(detectRt);
		//cv::Mat detectGray_crop;
		//cv::copyTo(detectGray, detectGray_crop, det_region_crop);
		//detectGray = detectGray_crop;

		//kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));
		//cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		//cv::morphologyEx(detectGray, detectGray, cv::MORPH_CLOSE, kernel);
		//cv::morphologyEx(detectGray, detectGray, cv::MORPH_OPEN, open_kernel);
		connected_num = 0;
		try
		{
			connected_num = cv::connectedComponentsWithStats(dst, labels, stats, centroids);
		}
		catch (const std::exception&)
		{
			spdlog::get("CATL_WCP")->error("无连通域");
			return;
		}
		//maxAreaLabel = 0;
		maxArea = 0;
		DefectData temp_defect_qipao;
		temp_defect_qipao.defect_id = 19;
		temp_defect_qipao.defect_name = "22_JiaoZhiQiPao";
		temp_defect_qipao.score = 0.7;
		for (int i = 1; i < connected_num; i++)
		{
			int qipao_x = stats.at<int>(i, 0);
			int qipao_y = stats.at<int>(i, 1);
			int qipao_w = stats.at<int>(i, 2);
			int qipao_h = stats.at<int>(i, 3);


			if (qipao_x == 0)
			{
				continue;
			}

			//区分气泡和打皱20241028
			//int _x = (qipao_x - 10 <= 0)? 0: qipao_x - 10;
			int _x = (qipao_x - 10 <= 0) ? 0 : qipao_x - 10; //打皱
			int _y = (qipao_y - 10 <= 0) ? 0 : qipao_y - 10; //打皱
			int x_ = (qipao_x + qipao_w + 20 <= dst.cols) ? qipao_x + qipao_w + 20 : dst.cols;
			int y_ = (qipao_y + qipao_h + 20 <= dst.rows) ? qipao_y + qipao_h + 20 : dst.rows;
			cv::Mat defect = dst(cv::Rect(_x, _y, x_ - _x, y_ - _y)); //打皱
			//cv::Mat defect = detectGray(cv::Rect(qipao_x - 10, qipao_y - 10, qipao_w + 20, qipao_h + 20));
			std::vector<std::vector<cv::Point>> contours2;
			cv::findContours(defect, contours2, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			double area = cv::contourArea(contours2[0]);
			cv::RotatedRect r1 = cv::minAreaRect(contours2[0]);
			float cz = 0, dz = 0;
			if (r1.size.width > r1.size.height)
			{
				cz = r1.size.width;
				dz = r1.size.height;
			}
			else
			{
				dz = r1.size.width;
				cz = r1.size.height;
			}

			if (cz / dz > 10) // 20_JiaoZhiDaZhou
			{
				continue;
				/*temp_defect_qipao.defect_id = 20;
				temp_defect_qipao.defect_name = "20_JiaoZhiDaZhou";

				if (abs(qipao_y + tiejiaoRt.y + detectRt.y - y1) < abs(qipao_y + tiejiaoRt.y + detectRt.y - y2))
				{
					dazhou_dis = abs(qipao_y + tiejiaoRt.y + detectRt.y - y1);
				}
				else
				{
					dazhou_dis = abs(qipao_y + tiejiaoRt.y + detectRt.y + qipao_h - y1);
				}

				if (dazhou_dis > dazhou_disthre)
				{
					continue;
				}*/

			}
			else // 22_JiaoZhiQiPao
			{
				temp_defect_qipao.defect_id = 22;
				temp_defect_qipao.defect_name = "22_JiaoZhiQiPao";
				//temp_defect_qipao.defect_name = "20_JiaoZhiDaZhou";
			}

			if (qipao_w <= qp_width_min || qipao_h <= qp_height_min)
			{
				continue;
			}


			temp_defect_qipao.x = stats.at<int>(i, 0) + img_x - piyi;
			temp_defect_qipao.y = stats.at<int>(i, 1) + y1;
			temp_defect_qipao.w = stats.at<int>(i, 2);
			temp_defect_qipao.h = stats.at<int>(i, 3);
			spdlog::get("CATL_WCP")->info("检测到" + temp_defect_qipao.defect_name);
			spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", defect_id:" + std::to_string(temp_defect_qipao.defect_id)
				+ ", x:" + std::to_string(temp_defect_qipao.x) + ", y:" + std::to_string(temp_defect_qipao.y) + ", w:" + std::to_string(temp_defect_qipao.w)
				+ ", h:" + std::to_string(temp_defect_qipao.h) + ", area:" + std::to_string(area * 0.078 * 0.078) + "mm2");
			/*LogWriterFlush("检测到" + temp_defect_qipao.defect_name);
			LogWriterFlush("defect_name:" + temp_defect_qipao.defect_name +", defect_id:" + std::to_string(temp_defect_qipao.defect_id)
			+ ", x:" + std::to_string(temp_defect_qipao.x) + ", y:" + std::to_string(temp_defect_qipao.y) + ", w:" + std::to_string(temp_defect_qipao.w)
			+ ", h:" + std::to_string(temp_defect_qipao.h));*/

			defect_data.emplace_back(temp_defect_qipao);
			//defect_data_result.emplace_back(temp_defect_qipao);
		}
		// 从标签映射中提取最大连通域
		//cv::Mat largestCC;
		//labels.copyTo(largestCC);
		//largestCC.setTo(0, labels != maxAreaLabel);
		//largestCC.setTo(255, labels == maxAreaLabel);
		//boundRect = cv::Rect(x + r_tiejiao.x, y + r_tiejiao.y, w, h);


		//defect_data.emplace_back(temp_defect);
		//defect_data_result.emplace_back(temp_defect);

		//continue;
	}
	catch (const std::exception& e)
	{
		spdlog::get("CATL_WCP")->error("传统算法检测气泡失败");
		spdlog::get("CATL_WCP")->error(std::string(e.what()));
		//LogWriterFlush("传统检测未检测到气泡打皱");
	}
}

bool compareContourAreas(std::vector<cv::Point> contour1, std::vector<cv::Point> contour2) {
	double i = fabs(contourArea(cv::Mat(contour1)));
	double j = fabs(contourArea(cv::Mat(contour2)));
	return (i > j);
}
void DetectDaZhou(cv::Mat& srcImg, std::vector<DefectData>& defect_data, int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, std::map<std::string, float>& m_defect_thresh_map, int detectType) {


	int qp_width_min = 0;
	int qp_height_min = 0;
	float dazhou_disthre = 0.5 / 0.078;
	int area_dazhou_min = 10;
	dazhou_disthre = m_defect_thresh_map["dazhou_disthre"] / 0.078;
	qp_width_min = m_defect_thresh_map["qp_width_min"] / 0.078;
	qp_height_min = m_defect_thresh_map["qp_height_min"] / 0.078;
	area_dazhou_min = m_defect_thresh_map["area_dazhou_min"] / 0.078 / 0.078;
	float dazhou_dis = 0;

	float dazhou_area = 0.5;
	float dazhou_gray = 50;
	float dazhou_cdb = 2.5;
	float dazhou_cz1 = 30;
	float dazhou_cz2 = 45;
	float dazhou_dz2 = 6;
	float dazhou_in_radius = 7;
	float dazhou_cz3 = 60;
	float dazhou_dz3 = 20;

	float dazhou_dis_angle = 5;
	float dazhou_cz4 = 500;
	float dazhou_dz4 = 50;
	float dazhou_thre_min = 35;
	float dazhou_use_autothre = 1;

	if (m_defect_thresh_map.find("dazhou_use_autothre") != m_defect_thresh_map.end())
		dazhou_use_autothre = m_defect_thresh_map["dazhou_use_autothre"];
	if(m_defect_thresh_map.find("dazhou_thre_min") != m_defect_thresh_map.end())
		dazhou_thre_min = m_defect_thresh_map["dazhou_thre_min"];
	

	dazhou_area = m_defect_thresh_map["dazhou_area"];
	dazhou_gray = m_defect_thresh_map["dazhou_gray"];
	dazhou_cdb = m_defect_thresh_map["dazhou_cdb"];
	dazhou_cz1 = m_defect_thresh_map["dazhou_cz1"];
	dazhou_cz2 = m_defect_thresh_map["dazhou_cz2"];
	dazhou_dz2 = m_defect_thresh_map["dazhou_dz2"];
	dazhou_in_radius = m_defect_thresh_map["dazhou_in_radius"];
	dazhou_cz3 = m_defect_thresh_map["dazhou_cz3"];
	dazhou_dz3 = m_defect_thresh_map["dazhou_dz3"];

	dazhou_dis_angle = m_defect_thresh_map["dazhou_dis_angle"];
	dazhou_cz4 = m_defect_thresh_map["dazhou_cz4"];
	dazhou_dz4 = m_defect_thresh_map["dazhou_dz4"];

	try
	{
		//
		//创建上下2块检测区域
		std::vector<cv::Point> pointsUp;
		std::vector<cv::Point> pointsDown;
		int x_min=999999, x_max=0, y_min=999999, y_max=0;
		for (int i = 0; i < XYdataUp.size(); i++)
		{
			if (XYdataUp[i].first < x_min)
				x_min = XYdataUp[i].first;
			if (XYdataUp[i].first > x_max)
				x_max = XYdataUp[i].first;
			if (XYdataUp[i].second < y_min)
				y_min = XYdataUp[i].second;
			if (XYdataUp[i].second > y_max)
				y_max = XYdataUp[i].second;
			pointsUp.push_back(cv::Point(XYdataUp[i].first, XYdataUp[i].second));
		}
		for (int i = 0; i < XYdataDown.size(); i++)
		{
			if (XYdataDown[i].first < x_min)
				x_min = XYdataDown[i].first;
			if (XYdataDown[i].first > x_max)
				x_max = XYdataDown[i].first;
			if (XYdataDown[i].second < y_min)
				y_min = XYdataDown[i].second;
			if (XYdataDown[i].second > y_max)
				y_max = XYdataDown[i].second;
			pointsDown.push_back(cv::Point(XYdataDown[i].first, XYdataDown[i].second));
		}


		cv::Mat mask_regionUp(srcImg.size(), CV_8UC1, cv::Scalar(0));
		cv::Mat mask_regionDown(srcImg.size(), CV_8UC1, cv::Scalar(0));
		cv::fillPoly(mask_regionUp, pointsUp, cv::Scalar(255));
		cv::fillPoly(mask_regionDown, pointsDown, cv::Scalar(255));
		

		cv::Rect tiejiaoRt(x_min, y_min, x_max - x_min, y_max - y_min);
		cv::Mat tiejiao_crop = srcImg(tiejiaoRt).clone();
		std::vector<cv::Mat> BGR;
		cv::split(tiejiao_crop, BGR);
		cv::Mat det_regionUp = mask_regionUp(tiejiaoRt);
		cv::Mat det_regionDown = mask_regionDown(tiejiaoRt);
		cv::Mat tiejiao_src = BGR[2].clone();

		std::vector<cv::Mat> regions;
		regions.push_back(det_regionUp);
		regions.push_back(det_regionDown);

		for (int i = 0; i < regions.size(); i++)
		{
			cv::Mat mask_region = regions[i];
			////对检测区域腐蚀3个像素
			cv::Mat kernel55 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
			cv::morphologyEx(mask_region, mask_region, cv::MORPH_ERODE, kernel55); // 


			vector<vector<cv::Point>> contours1;
			findContours(mask_region, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			// 遍历所有轮廓，找到最小的包围盒
			cv::Rect boundingRect;
			bool first = true;
			for (const auto& contour : contours1) {
				cv::Rect rect = cv::boundingRect(contour);
				if (first || (rect.area() < boundingRect.area() && rect.area() > 0)) {
					boundingRect = rect;
					first = false;
				}
			}


			cv::Mat det_src = tiejiao_src(boundingRect);
			
			cv::Mat rest;
			//拿到缺陷区域
			if (dazhou_use_autothre)
			{
				//自适应阈值分割
				cv::adaptiveThreshold(det_src, rest, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY, 61, 2);
			}
			else
			{
				cv::threshold(det_src, rest, dazhou_thre_min, 255, cv::THRESH_BINARY);
			}


			

			//把属于检测区域的缺陷抠出来
			/*cv::Mat rest;
			cv::copyTo(tiejiao_crop, rest, det_region);*/

			cv::Mat labels, centroids, stats, res_img;
			int connected_num = 0;
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(rest, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
			cv::Mat dst = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
			double maxArea = 0;
			int maxIdx = 0;
			//int areaMin = 10000;
			int areaMin = 10, areaMax = 40000000;
			for (int i = 0; i < contours.size(); i++) {
				double area = cv::contourArea(contours[i]);
				//if (area > areaMin) {
				if (area > area_dazhou_min && area < areaMax) {

					maxArea = area;
					maxIdx = i;
					cv::drawContours(dst, contours, static_cast<int>(maxIdx), cv::Scalar(255), -1, 1);
				}
			}

			int maxAreaLabel = 0;
			maxArea = 0;
			int x, y, w, h;
			//int x_min = 9999;




			connected_num = 0;
			try
			{
				connected_num = cv::connectedComponentsWithStats(dst, labels, stats, centroids);
			}
			catch (const std::exception&)
			{
				spdlog::get("CATL_WCP")->error("无连通域");
				return;
			}
			maxAreaLabel = 0;
			maxArea = 0;
			DefectData temp_defect_qipao;
			temp_defect_qipao.defect_id = 20;
			temp_defect_qipao.defect_name = "20_JiaoZhiDaZhou";
			temp_defect_qipao.score = 0.7;
			for (int i = 1; i < connected_num; i++)
			{
				int qipao_x = stats.at<int>(i, 0);
				int qipao_y = stats.at<int>(i, 1);
				int qipao_w = stats.at<int>(i, 2);
				int qipao_h = stats.at<int>(i, 3);


				if (qipao_x == 0)
				{
					continue;
				}

				//裁切单个缺陷以计算特征值，用来过滤
				int _x = (qipao_x - 5 <= 0) ? 0 : qipao_x - 5; //打皱
				int _y = (qipao_y - 5 <= 0) ? 0 : qipao_y - 5; //打皱
				int x_ = (qipao_x + qipao_w + 10 <= dst.cols - 1) ? qipao_x + qipao_w + 10 : dst.cols - 1;
				int y_ = (qipao_y + qipao_h + 10 <= dst.rows - 1) ? qipao_y + qipao_h + 10 : dst.rows - 1;
				cv::Mat defect = dst(cv::Rect(_x, _y, x_ - _x, y_ - _y)); //打皱
				cv::Mat defect_gray = tiejiao_src(cv::Rect(_x, _y, x_ - _x, y_ - _y));
				
				
				
				std::vector<std::vector<cv::Point>> contours2;
				cv::findContours(defect, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				//轮廓按面积排序
				std::sort(contours.begin(), contours.end(), compareContourAreas);
				if (contours.size() < 1)
					continue;
				double area = cv::contourArea(contours[0]);
				cv::RotatedRect r1 = cv::minAreaRect(contours[0]);

				//长轴，短轴，灰度
				float cz = 0, dz = 0;
				if (r1.size.width > r1.size.height)
				{
					cz = r1.size.width;
					dz = r1.size.height;
				}
				else
				{
					dz = r1.size.width;
					cz = r1.size.height;
				}

				//斜率
				float angle = r1.angle;

				//最大内接圆半径
				double radius;
				cv::Mat1f dt;
				cv::distanceTransform(defect, dt, cv::DIST_L2, 0, cv::DIST_LABEL_PIXEL);

				// Find max value
				double max_val;
				cv::Point max_loc;
				cv::minMaxLoc(dt, nullptr, &max_val, nullptr, &max_loc);

				// Output image
				/*cv::Mat out;
				cv::cvtColor(defect, out, cv::COLOR_GRAY2BGR);
				cv::circle(out, max_loc, max_val, cv::Scalar(0, 255, 0));*/
				radius = max_val;

				//计算平均灰度
				cv::Scalar graymean = cv::mean(defect_gray, defect);
				double gray_m = graymean[0];






				//面积和灰度过滤
				float area_mm = area * 0.078 * 0.078;
				if (area_mm < dazhou_area && gray_m < dazhou_gray)
				{
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", area:" + std::to_string(area * 0.078 * 0.078) + "mm2 < 阈值:" + std::to_string(dazhou_area) + ", 灰度:" + std::to_string(dazhou_gray)
						+ " < 阈值" + std::to_string(dazhou_gray) + ",被过滤");
					continue;
				}

				//过滤小点点
				float cdb = cz / dz;
				if (cdb < dazhou_cdb && cz < dazhou_cz1)
				{
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", 长轴像素:" + std::to_string(cz) + " < 阈值 : " + std::to_string(dazhou_cz1) + ", 长短轴比:" + std::to_string(cdb)
						+ " < 阈值" + std::to_string(dazhou_cdb) + ",被过滤");
					continue;
				}
				//过滤边缘
				/*if (dz < dazhou_dz2 && cz < dazhou_cz2)
				{
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", 长轴像素:" + std::to_string(cz) + " < 阈值 : " + std::to_string(dazhou_cz2) + ", 短轴像素:" + std::to_string(dz)
						+ " < 阈值" + std::to_string(dazhou_dz2) + ",被过滤");
					continue;
				}*/
				if (dz < dazhou_dz3 && cz < dazhou_cz3 && radius > dazhou_in_radius)
				{
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", 长轴像素:" + std::to_string(cz) + " < 阈值 : " + std::to_string(dazhou_cz3) + ", 短轴像素:" + std::to_string(dz)
						+ " < 阈值" + std::to_string(dazhou_dz3) + ", 最大内接圆半径:" + std::to_string(radius) + " > 阈值 : " + std::to_string(dazhou_in_radius) + ",被过滤");
					continue;
				}

				if ((abs(90 - angle) < dazhou_dis_angle || abs(0 - angle) < dazhou_dis_angle) && cz < dazhou_cz4 && dz < dazhou_dz4)
				{
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", 长轴像素:" + std::to_string(cz) + " < 阈值 : " + std::to_string(dazhou_cz4) + ", 短轴像素:" + std::to_string(dz)
						+ " < 阈值" + std::to_string(dazhou_dz4) + ", 角度:" + std::to_string(angle) + " 与水平线差异 < 阈值 : " + std::to_string(dazhou_dis_angle) + ",被过滤");
					continue;
				}


				temp_defect_qipao.x = stats.at<int>(i, 0) + x_min + boundingRect.x;
				temp_defect_qipao.y = stats.at<int>(i, 1) + y_min + boundingRect.y;
				temp_defect_qipao.w = stats.at<int>(i, 2);
				temp_defect_qipao.h = stats.at<int>(i, 3);
				spdlog::get("CATL_WCP")->info("检测到" + temp_defect_qipao.defect_name);
				spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", defect_id:" + std::to_string(temp_defect_qipao.defect_id)
					+ ", x:" + std::to_string(temp_defect_qipao.x) + ", y:" + std::to_string(temp_defect_qipao.y) + ", w:" + std::to_string(temp_defect_qipao.w)
					+ ", h:" + std::to_string(temp_defect_qipao.h) + ", area:" + std::to_string(area_mm) + "mm2" + ", 灰度:" + std::to_string(dazhou_gray)
					+ ", 长轴像素:" + std::to_string(cz) + ", 短轴像素:" + std::to_string(dz) + ", 长短轴比:" + std::to_string(cdb) + ", 最大内接圆半径:" + std::to_string(radius)
					+ ", 角度:" + std::to_string(angle));



				defect_data.emplace_back(temp_defect_qipao);
		}

		
			
			
		}
		
	}
	catch (const std::exception& e)
	{
		spdlog::get("CATL_WCP")->error("传统算法检测打皱失败");
		spdlog::get("CATL_WCP")->error(std::string(e.what()));
		//LogWriterFlush("传统检测未检测到气泡打皱");
	}
}

StatusCode Detection_J::Detecting(cv::Mat& img, std::vector<DefectData>& defect_data, int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, int tape_flag)
{
	//double time1;  // 检测时间记录
	//double t_cost;  // 检测时间统计

	// time1 = static_cast<double>(cv::getTickCount());
	if (img.empty()) {
		spdlog::get("CATL_WCP")->error("read image failed!");
		//LogWriterFlush("read image failed!.");
		return StatusCode::ROI_IMG_EMPTY;
	}
	if (img.rows < 320 || img.cols < 320) {
		spdlog::get("CATL_WCP")->error("image height<320||width<320");
		//LogWriterFlush("image height<320||width<320!.");
		return StatusCode::ROI_IMG_EMPTY;
	}
	//defect_data.clear();
	defect_data_result.clear();
	double ratio;
	/*double res_w = img.cols / 640.00;
	double res_h = img.rows / 640.00;*/
	//2024/11/2 lil17 供应商传入的img_x坐标为级片边缘+10，修改为AT11边缘
	cv::Rect r1 = cv::Rect(img_x - piyi, 0, img_w, img.rows - 1);
	cv::Mat srcImg;
	img(r1).copyTo(srcImg);
	//cv::resize(img, img, cv::Size(640, 640));
	auto t1 = std::chrono::steady_clock::now();
	cv::Mat loujinshu_test, loujinshu_test_bgr;
	cv::Mat labels_, centroids_, stats_;
	cv::Mat carch;
	

	cv::Mat pr_img = preprocess_img(srcImg, INPUT_W, INPUT_H, &ratio); //  BGR to RGB
	int i = 0;
	double all_area_loujinshu = 0;
	for (int row = 0; row < INPUT_H; ++row) {
		uchar* uc_pixel = pr_img.data + row * pr_img.step;
		for (int col = 0; col < INPUT_W; ++col) {
			data[i] = (float)uc_pixel[2] / 255.0;
			data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
			data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
			uc_pixel += 3;
			++i;
		}
	}

	/*t_cost = (static_cast<double>(cv::getTickCount()) - time1) / cv::getTickFrequency() * 1000;
	std::cout << "front cost(ms): " << t_cost << std::endl;*/

	std::vector<Yolo::Detection> batch_res;

	// time1 = static_cast<double>(cv::getTickCount());
	doInference(*context, stream, buffers, data, prob, BATCH_SIZE);	
	nms(batch_res, &prob[0], CONF_THRESH, NMS_THRESH);

#pragma region MyRegion
	/*t_cost = (static_cast<double>(cv::getTickCount()) - time1) / cv::getTickFrequency() * 1000;
	LogWriterFlush("模型推理时间" + std::to_string(t_cost) + "ms");*/
	//std::cout << "doinference cost(ms): " << t_cost << std::endl;


	// time1 = static_cast<double>(cv::getTickCount());
	//cv::Mat posunMat;
	//std::vector<cv::Mat> BGR;
	//cv::split(img, BGR);
	//cv::threshold(BGR[0], posunMat, 200, 255, cv::THRESH_BINARY);
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 50));
	//cv::morphologyEx(posunMat, posunMat, cv::MORPH_OPEN, kernel);
	//posunMat.convertTo(posunMat, CV_8UC1);
	//cv::Mat labels, centroids, stats, res_img;
	//int connected_num = cv::connectedComponentsWithStats(posunMat, labels, stats, centroids);
	//// 找出面积最大的连通域
	//int maxAreaLabel = 0;
	//int maxArea = 0;
	//int x, y, w, h;
	//for (int i = 1; i < connected_num; i++)
	//{
	//	int a = stats.at<int>(i, 4);
	//	if (a > maxArea) {
	//		maxArea = a;
	//		maxAreaLabel = i;
	//	}
	//	x = stats.at<int>(maxAreaLabel, 0);
	//	y = stats.at<int>(maxAreaLabel, 1);
	//	w = stats.at<int>(maxAreaLabel, 2);
	//	h = stats.at<int>(maxAreaLabel, 3);
	//}
	//cv::Rect boundRectAT(0, 0, 0, 0);
	//// boundRectAT = cv::Rect(x, y, w, h);
	//boundRectAT = cv::Rect(x + w, 0, img.cols - x - w, img.rows);
	//cv::Mat jipianMat = img(boundRectAT);
	//cv::split(jipianMat, BGR);
	//cv::Mat jiance1, jiance2;
	//cv::Mat oriMat = BGR[2];
	//cv::Rect jiance1Rect(3, 0, 20, jipianMat.rows);
	//cv::Rect jiance2Rect(jipianMat.cols - 25, 0, 24, jipianMat.rows);
	//jiance1 = (BGR[2])(jiance1Rect);
	//jiance2 = (BGR[2])(jiance2Rect);

	//cv::Mat quexian1, quexian2;
	//cv::threshold(jiance1, quexian1, 25, 255, cv::THRESH_BINARY_INV);
	//cv::threshold(jiance2, quexian2, 20, 255, cv::THRESH_BINARY_INV);
	//connected_num = cv::connectedComponentsWithStats(quexian1, labels, stats, centroids);
	//std::vector<cv::Rect> posunRects1;
	//for (int i = 1; i < connected_num; i++)
	//{
	//	x = stats.at<int>(i, 0);
	//	y = stats.at<int>(i, 1);
	//	w = stats.at<int>(i, 2);
	//	h = stats.at<int>(i, 3);
	//	if (h < 8 || w < 5)
	//		continue;
	//	posunRects1.push_back(cv::Rect(x, y, w, h));
	//}
	///*for (size_t i = 0; i < posunRects1.size(); i++)
	//{
	//	cv::rectangle(jiance1, posunRects1[i], -1, 11);
	//}*/
	//connected_num = cv::connectedComponentsWithStats(quexian2, labels, stats, centroids);
	//std::vector<cv::Rect> posunRects2;
	//for (int i = 1; i < connected_num; i++)
	//{
	//	x = stats.at<int>(i, 0);
	//	y = stats.at<int>(i, 1);
	//	w = stats.at<int>(i, 2);
	//	h = stats.at<int>(i, 3);
	//	if (h < 8 || w < 5)
	//		continue;
	//	posunRects2.push_back(cv::Rect(x, y, w, h));
	//}
	///*for (size_t i = 0; i < posunRects2.size(); i++)
	//{
	//	cv::rectangle(jiance2, posunRects2[i], -1, 11);
	//}*/
	//
	//DefectData temp_defect_posun;
	//temp_defect_posun.defect_name == "01_JiPianPoSun";
	//for (size_t i = 0; i < posunRects1.size(); i++)
	//{
	//	// temp_defect_posun.defect_name == "01_JiPianPoSun";
	//	temp_defect_posun.score = 0.9;
	//	temp_defect_posun.defect_id = 1;
	//	temp_defect_posun.x = (posunRects1[i].x + jiance1Rect.x + boundRectAT.x) * res_w;
	//	temp_defect_posun.y = (posunRects1[i].y + jiance1Rect.y) * res_h;
	//	temp_defect_posun.h = posunRects1[i].height * res_h;
	//	temp_defect_posun.w = posunRects1[i].width * res_w;
	//	temp_defect_posun.defect_name == "01_JiPianPoSun";
	//	defect_data.emplace_back(temp_defect_posun);
	//	defect_data_result.emplace_back(temp_defect_posun);
	//}
	//for (size_t i = 0; i < posunRects2.size(); i++)
	//{
	//	temp_defect_posun.defect_name == "01_JiPianPoSun";
	//	temp_defect_posun.score = 0.9;
	//	temp_defect_posun.defect_id = 1;
	//	temp_defect_posun.x = (posunRects2[i].x + jiance2Rect.x + boundRectAT.x) * res_w;
	//	temp_defect_posun.y = (posunRects2[i].y + jiance2Rect.y) * res_h;
	//	temp_defect_posun.h = posunRects2[i].height * res_h;
	//	temp_defect_posun.w = posunRects2[i].width * res_w;
	//	defect_data.emplace_back(temp_defect_posun);
	//	defect_data_result.emplace_back(temp_defect_posun);
	//}



	// 右侧极片破损检测 0724

	//if (m_defect_thresh_map["01_JiPianPoSun"] < 0.75)
	//{
	//	int crop_width = 30;
	//	cv::Rect right_crop_rt(img.cols - crop_width, 0, crop_width, img.rows - 1);
	//	cv::Mat posun_img;
	//	posun_img = img(right_crop_rt);
	//	cv::cvtColor(posun_img, posun_img, cv::COLOR_BGR2GRAY);
	//	int result_mean_right = cv::mean(posun_img).val[0];
	//	cv::Mat posun_img_result;
	//	cv::threshold(posun_img, posun_img_result, result_mean_right*1.5, 255, cv::THRESH_BINARY);
	//	//cv::inRange(posun_img, result_mean_right * 1.5, 120, posun_img_result);
	//	cv::Mat kernel_pos = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	//	cv::morphologyEx(posun_img_result, posun_img_result, cv::MORPH_OPEN, kernel_pos);
	//	cv::Mat labels, centroids, stats;
	//	int connected_num = cv::connectedComponentsWithStats(posun_img_result, labels, stats, centroids);
	//	int x, y, w, h;
	//	std::vector<cv::Rect> posunRect;
	//	for (int i = 1; i < connected_num; i++)
	//	{
	//		x = stats.at<int>(i, 0);
	//		y = stats.at<int>(i, 1);
	//		w = stats.at<int>(i, 2);
	//		h = stats.at<int>(i, 3);
	//		if (w < 2 || h < 2 || h > 100)
	//			continue;
	//		posunRect.push_back(cv::Rect(x, y, w, h));
	//	}
	//	DefectData temp_defect_posun;
	//	temp_defect_posun.defect_name = "01_JiPianPoSun";
	//
	//	for (size_t i = 0; i < posunRect.size(); i++)
	//	{
	//		// temp_defect_posun.defect_name = "01_JiPianPoSun";
	//		temp_defect_posun.score = 0.72;
	//		temp_defect_posun.defect_id = 1;
	//		temp_defect_posun.x = (posunRect[i].x + right_crop_rt.x ) * res_w;
	//		temp_defect_posun.y = (posunRect[i].y + right_crop_rt.y) * res_h;
	//		temp_defect_posun.h = posunRect[i].height * res_h;
	//		temp_defect_posun.w = posunRect[i].width * res_w;
	//		//temp_defect_posun.defect_name = classes[1];;
	//		defect_data.emplace_back(temp_defect_posun);
	//		defect_data_result.emplace_back(temp_defect_posun);
	//	}
	//}



	//漏金属传统算法部分
	//
	//if (loujinshu_test.channels() == 3)
	//	cv::cvtColor(loujinshu_test, loujinshu_test, cv::COLOR_RGB2GRAY);
	//cv::threshold(loujinshu_test, loujinshu_test, 80, 255, cv::THRESH_BINARY);
	//cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
	//cv::morphologyEx(loujinshu_test, loujinshu_test, cv::MORPH_CLOSE, kernel);
	//loujinshu_test.convertTo(loujinshu_test, CV_8UC1);
	//int connected_num = cv::connectedComponentsWithStats(loujinshu_test, labels_, stats_, centroids_);
	//for (int i = 1; i < connected_num; ++i)
	//{
	//	if ((stats_.at<int>(i, cv::CC_STAT_WIDTH) < img.cols/5 && img.rows/2 < stats_.at<int>(i, cv::CC_STAT_HEIGHT)))
	//		continue;
	//	if ((stats_.at<int>(i, cv::CC_STAT_HEIGHT) > 30 || stats_.at<int>(i, cv::CC_STAT_WIDTH) > 30))
	//	//if ((stats_.at<int>(i, cv::CC_STAT_HEIGHT) > 10 || stats_.at<int>(i, cv::CC_STAT_WIDTH) > 10))
	//	{
	//		loujinshu_test_bgr = img(cv::Rect(img.cols / 10 + stats_.at<int>(i, cv::CC_STAT_LEFT), stats_.at<int>(i, cv::CC_STAT_TOP), stats_.at<int>(i, cv::CC_STAT_WIDTH), stats_.at<int>(i, cv::CC_STAT_HEIGHT)));
	//		double minval=0.0, maxval=0.0;
	//		if (loujinshu_test_bgr.channels() == 3)
	//			cv::cvtColor(loujinshu_test_bgr, loujinshu_test_bgr, cv::COLOR_RGB2GRAY);
	//		cv::minMaxLoc(loujinshu_test_bgr, &minval, &maxval);
	//		if (maxval > 180)
	//		{
	//			temp_defect1.defect_name = "12_LouJinShu";
	//			//temp_defect1.defect_name = "01_JiPianPoSun";
	//			temp_defect1.score = 0.6;
	//			temp_defect1.defect_id = 12;
	//			//temp_defect1.defect_id = 1;
	//			temp_defect1.x = (stats_.at<int>(i, cv::CC_STAT_LEFT) + img.cols / 10) * res_w;
	//			temp_defect1.y = stats_.at<int>(i, cv::CC_STAT_TOP) * res_h;
	//			temp_defect1.h = stats_.at<int>(i, cv::CC_STAT_HEIGHT) * res_h;
	//			temp_defect1.w = (stats_.at<int>(i, cv::CC_STAT_WIDTH)) * res_w;
	//			defect_data.emplace_back(temp_defect1);
	//			defect_data_result.emplace_back(temp_defect1);
	//		}
	//	}
	//}
#pragma endregion

	

	DefectData temp_defect, temp_defect1;
	bool fanguang = false;
	cv::Rect r_tiejiao,r_tiejiao1;
	cv::Rect boundRect(0, 0, 0, 0);
	cv::Rect r_tiejiao_center;
	int result_flag = 0;
	int result_num = batch_res.size();

	int qp_width_min = 0;
	int qp_height_min = 0;
	float dazhou_disthre = 0.5 / 0.078;
	dazhou_disthre = m_defect_thresh_map["dazhou_disthre"] / 0.078;
	qp_width_min = m_defect_thresh_map["qp_width_min"] / 0.078;
	qp_height_min = m_defect_thresh_map["qp_height_min"] / 0.078;
	float dazhou_dis = 0;

	//// 贴胶定位传统算法----------------------------------------------------------------------------------------------------------------
	//std::vector<cv::Mat> BGR_ORI;
	//cv::split(srcImg, BGR_ORI);
	//cv::Mat tiejiao_mat;
	//cv::inRange(BGR_ORI[2], 80, 100, tiejiao_mat);
	//cv::Mat tiejiao_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));

	////cv::morphologyEx(temp, temp, cv::MORPH_OPEN, kernel);
	//cv::morphologyEx(tiejiao_mat, tiejiao_mat, cv::MORPH_CLOSE, tiejiao_kernel);

	//cv::Mat tiejiao_labels, tiejiao_centroids, tiejiao_stats, tiejiao_res_img;
	//int connected_num_tiejiao;
	//try
	//{
	//	connected_num_tiejiao = cv::connectedComponentsWithStats(tiejiao_mat, tiejiao_labels, tiejiao_stats, tiejiao_centroids);
	//}
	//catch (const std::exception&)
	//{
	//	LogWriterFlush("无连通域");
	//}

	//int tiejiao_maxarea = 0;
	//int tiejiao_i;
	//int tiejiao_x=0, tiejiao_y=0, tiejiao_w=0, tiejiao_h=0;
	//for (int i = 1; i < connected_num_tiejiao; i++)
	//{
	//	int a = tiejiao_stats.at<int>(i, 4);
	//	int temp_w = tiejiao_stats.at<int>(i, 2);
	//	if (a > tiejiao_maxarea && temp_w >1000) {
	//		tiejiao_maxarea = a;
	//		tiejiao_i = i;
	//	}
	//	else
	//	{
	//		continue;
	//	}
	//	tiejiao_x = tiejiao_stats.at<int>(tiejiao_i, 0);
	//	tiejiao_y = tiejiao_stats.at<int>(tiejiao_i, 1);
	//	tiejiao_w = tiejiao_stats.at<int>(tiejiao_i, 2);
	//	tiejiao_h = tiejiao_stats.at<int>(tiejiao_i, 3);
	//}
	



	


	for (size_t js = 0; js < batch_res.size(); js++) 
	{
		
		temp_defect.defect_name = classes[(int)batch_res[js].class_id];
		// if (temp_defect.defect_name == "05_OK_Tiejiao_B" || temp_defect.defect_name == "07_OK_Tiejiao_Y" || temp_defect.defect_name == "08_OK_LenLieWen")
		
		if (temp_defect.defect_name == "05_OK_Tiejiao_B" || temp_defect.defect_name == "07_OK_Tiejiao_Y")
		{
			if (r_tiejiao.width>0)
				r_tiejiao1 = get_rect(img, batch_res[js].bbox);
			r_tiejiao = get_rect(img, batch_res[js].bbox);

			if (temp_defect.defect_name == "05_OK_Tiejiao_B")
			{
				continue;
			}

#pragma region MyRegion

			// JC L10 胶纸打皱过滤 ------

			//if (result_num >= 2)
			//{
			//	cv::Mat temp;
			//	std::vector<cv::Mat> BGR;
			//	temp = img(r_tiejiao);
			//	cv::split(temp, BGR);
			//	if (temp_defect.defect_name == "07_OK_Tiejiao_Y")
			//	{
			//		cv::threshold(BGR[2], temp, 100, 255, cv::THRESH_BINARY);
			//	}
			//	else if (temp_defect.defect_name == "07_OK_Tiejiao_B")
			//	{
			//		cv::threshold(BGR[0], temp, 120, 255, cv::THRESH_BINARY);
			//	}
			//	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			//	cv::morphologyEx(temp, temp, cv::MORPH_OPEN, kernel);
			//	//temp.convertTo(temp, CV_8UC1);
			//	cv::Mat labels, centroids, stats, res_img;
			//	int connected_num = cv::connectedComponentsWithStats(temp, labels, stats, centroids);
			//	// 找出面积最大的连通域
			//	int maxAreaLabel = 0;
			//	int maxArea = 0;
			//	int x, y, w, h;

			//	for (int i = 1; i < connected_num; i++)
			//	{
			//		int a = stats.at<int>(i, 4);
			//		if (a > maxArea) {
			//			maxArea = a;
			//			maxAreaLabel = i;
			//		}
			//		x = stats.at<int>(maxAreaLabel, 0);
			//		y = stats.at<int>(maxAreaLabel, 1);
			//		w = stats.at<int>(maxAreaLabel, 2);
			//		h = stats.at<int>(maxAreaLabel, 3);
			//	}
			//	// 从标签映射中提取最大连通域
			//	/*cv::Mat largestCC;
			//	labels.copyTo(largestCC);
			//	largestCC.setTo(0, labels != maxAreaLabel);
			//	largestCC.setTo(255, labels == maxAreaLabel);*/
			//	boundRect = cv::Rect(x + r_tiejiao.x, y + r_tiejiao.y, w, h);
			//}

/*
			// HX L50 胶纸气泡打皱传统算法部分------在AI定位到贴胶后进入下面逻辑------------------------------------------------------
			try
			{
				//cv::Mat temp;
				std::vector<cv::Mat> BGR;

				int tj_x, tj_y, tj_w, tj_h;
				//tj_x = r_tiejiao.x * res_w;
				//tj_y = r_tiejiao.y * res_h;
				//tj_h = r_tiejiao.height * res_h;
				//tj_w = r_tiejiao.width * res_w;
				//cv::Rect tiejiaoRt(tj_x, tj_y, tj_w, tj_h);
				//cv::Mat temp = srcImg(tiejiaoRt);

				cv::Rect tiejiaoRt(r_tiejiao.x, r_tiejiao.y, r_tiejiao.width, r_tiejiao.height);
				cv::Mat tiejiao_crop = srcImg(tiejiaoRt).clone();

				cv::split(tiejiao_crop, BGR);

				////**2024/10/25lilu**抓到胶纸打皱与气泡需要检测的区域
				cv::Mat det_bin;
				cv::threshold(BGR[2], det_bin, 30, 255, cv::THRESH_BINARY_INV);
				cv::Mat kernel_jiao = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 10));
				cv::morphologyEx(det_bin, det_bin, cv::MORPH_CLOSE, kernel_jiao);
				std::vector<std::vector<cv::Point>> contours1;
				cv::findContours(det_bin, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

				cv::Mat det_region = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
				for (int i = 0; i < contours1.size(); i++) {
					double area = cv::contourArea(contours1[i]);
					//if (area > areaMin) {
					if (area > 100000) {

						cv::drawContours(det_region, contours1, static_cast<int>(i), cv::Scalar(255), -1, 1);
					}
				}


				//拿到缺陷区域
				cv::threshold(BGR[2], tiejiao_crop, 35, 255, cv::THRESH_BINARY);
				cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));				
				cv::morphologyEx(tiejiao_crop, tiejiao_crop, cv::MORPH_CLOSE, kernel);
				//kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(20, 2));
				//cv::morphologyEx(tiejiao_crop, tiejiao_crop, cv::MORPH_OPEN, kernel);
				
				//把属于检测区域的缺陷抠出来
				cv::Mat rest;
				cv::copyTo(tiejiao_crop, rest, det_region);
				kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
				cv::morphologyEx(rest, rest, cv::MORPH_OPEN, kernel);
				
				cv::Mat labels, centroids, stats, res_img;
				int connected_num = 0;
				std::vector<std::vector<cv::Point>> contours;
				cv::findContours(rest, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				cv::Mat dst = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
				double maxArea = 0;
				int maxIdx = 0;
				//int areaMin = 10000;
				int areaMin = 1000, areaMax = 20000;
				for (int i = 0; i < contours.size(); i++) {
					double area = cv::contourArea(contours[i]);
					//if (area > areaMin) {
					if (area > areaMin && area < areaMax) {
						maxArea = area;
						maxIdx = i;
						cv::drawContours(dst, contours, static_cast<int>(maxIdx), cv::Scalar(255), -1, 1);
					}
				}

				int maxAreaLabel = 0;
				maxArea = 0;
				int x, y, w, h;
				int x_min = 9999;

				////20241024胶纸打皱
				//cv::Mat dst1 = cv::Mat::zeros(tiejiao_crop.size(), CV_8UC1);
				//cv::Mat grayTemp1, grayTemp2;
				//cv::threshold(tiejiao_crop, grayTemp1, 200, 255, cv::THRESH_BINARY_INV);
				//

				//std::vector<std::vector<cv::Point>> contours1;
				//cv::findContours(grayTemp1, contours1, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
				//for (int i = 0; i < contours1.size(); i++) {
				//	double area = cv::contourArea(contours1[i]);
				//	//if (area > areaMin) {
				//	if (area > 100000) {

				//		cv::drawContours(dst1, contours1, static_cast<int>(i), cv::Scalar(255), -1, 1);
				//	}
				//}

				//cv::Mat kernel1 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(40, 40));
				//cv::morphologyEx(dst1, grayTemp2, cv::MORPH_CLOSE, kernel1);

				//cv::Mat rest;
				//cv::copyTo(tiejiao_crop, rest, grayTemp2);


				cv::Mat grayTemp;
				cv::threshold(tiejiao_crop, grayTemp, 200, 255, cv::THRESH_BINARY_INV);
				//cv::morphologyEx(grayTemp, grayTemp, cv::MORPH_OPEN, kernel);
				maxAreaLabel = 0;
				try
				{
					connected_num = cv::connectedComponentsWithStats(grayTemp, labels, stats, centroids);
				}
				catch (const std::exception&)
				{
					spdlog::get("CATL_WCP")->error("无连通域");
					//LogWriterFlush("无连通域");
					continue;
				}
			
				for (int i = 1; i < connected_num; i++)
				{
					int x_temp = stats.at<int>(i, 0);
					if (x_temp < x_min) {
						x_min = x_temp;
						maxAreaLabel = i;
					}
					x = stats.at<int>(maxAreaLabel, 0);
					y = stats.at<int>(maxAreaLabel, 1);
					w = stats.at<int>(maxAreaLabel, 2);
					h = stats.at<int>(maxAreaLabel, 3);
				}
				cv::Mat add_out;
				cv::add(BGR[2], dst, add_out);
			
				cv::Rect detectRt(x + w, 0, add_out.cols - x - w - 10, add_out.rows);
				cv::Mat detectImg = add_out(detectRt);
				cv::Mat detectGray;

				//cv::inRange(detectImg, 30, 254, detectGray);
				cv::inRange(detectImg, 240, 255, detectGray);

				kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, 2));
				cv::Mat open_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
				cv::morphologyEx(detectGray, detectGray, cv::MORPH_CLOSE, kernel);
				cv::morphologyEx(detectGray, detectGray, cv::MORPH_OPEN, open_kernel);
				connected_num = 0;
				try
				{
					connected_num = cv::connectedComponentsWithStats(detectGray, labels, stats, centroids);
				}
				catch (const std::exception&)
				{
					spdlog::get("CATL_WCP")->error("无连通域");
					continue;
				}
				maxAreaLabel = 0;
				maxArea = 0;
				DefectData temp_defect_qipao;
				temp_defect_qipao.defect_id = 19;
				temp_defect_qipao.defect_name = "22_JiaoZhiQiPao";
				temp_defect_qipao.score = 0.7;
				for (int i = 1; i < connected_num; i++)
				{
					int qipao_x = stats.at<int>(i, 0);
					int qipao_y = stats.at<int>(i, 1);
					int qipao_w = stats.at<int>(i, 2);
					int qipao_h = stats.at<int>(i, 3);


					if (qipao_x == 0)
					{
						continue;
					}

					//区分气泡和打皱20241028
					//int _x = (qipao_x - 10 <= 0)? 0: qipao_x - 10;
					cv::Mat defect = detectGray(cv::Rect(qipao_x - 10, qipao_y - 10, qipao_w + 20, qipao_h + 20));
					std::vector<std::vector<cv::Point>> contours2;
					cv::findContours(defect, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
					cv::RotatedRect r1 = cv::minAreaRect(contours[0]);
					float cz = 0, dz = 0;
					if (r1.size.width > r1.size.height)
					{
						cz = r1.size.width;
						dz = r1.size.height;
					}
					else
					{
						dz = r1.size.width;
						cz = r1.size.height;
					}
					
					if (cz / dz > 4) // 20_JiaoZhiDaZhou
					{
						temp_defect_qipao.defect_id = 20;
						temp_defect_qipao.defect_name = "20_JiaoZhiDaZhou";

						if (abs(qipao_y + tiejiaoRt.y + detectRt.y - y1) < abs(qipao_y + tiejiaoRt.y + detectRt.y - y2))
						{
							dazhou_dis = abs(qipao_y + tiejiaoRt.y + detectRt.y - y1);
						}
						else
						{
							dazhou_dis = abs(qipao_y + tiejiaoRt.y + detectRt.y + qipao_h - y1);
						}

						if (dazhou_dis > dazhou_disthre)
						{
							continue;
						}
						
					}
					else // 22_JiaoZhiQiPao
					{
						temp_defect_qipao.defect_id = 22;
						temp_defect_qipao.defect_name = "22_JiaoZhiQiPao";
						//temp_defect_qipao.defect_name = "20_JiaoZhiDaZhou";
					}

					if (qipao_w <= qp_width_min || qipao_h <= qp_height_min)
					{
						continue;
					}


					temp_defect_qipao.x = stats.at<int>(i, 0) + tiejiaoRt.x + detectRt.x;
					temp_defect_qipao.y = stats.at<int>(i, 1) + tiejiaoRt.y + detectRt.y;
					temp_defect_qipao.w = stats.at<int>(i, 2);
					temp_defect_qipao.h = stats.at<int>(i, 3);
					spdlog::get("CATL_WCP")->info("检测到" + temp_defect_qipao.defect_name);
					spdlog::get("CATL_WCP")->info("defect_name:" + temp_defect_qipao.defect_name + ", defect_id:" + std::to_string(temp_defect_qipao.defect_id)
						+ ", x:" + std::to_string(temp_defect_qipao.x) + ", y:" + std::to_string(temp_defect_qipao.y) + ", w:" + std::to_string(temp_defect_qipao.w)
						+ ", h:" + std::to_string(temp_defect_qipao.h));
					//LogWriterFlush("检测到" + temp_defect_qipao.defect_name);
					//LogWriterFlush("defect_name:" + temp_defect_qipao.defect_name +", defect_id:" + std::to_string(temp_defect_qipao.defect_id)
					//+ ", x:" + std::to_string(temp_defect_qipao.x) + ", y:" + std::to_string(temp_defect_qipao.y) + ", w:" + std::to_string(temp_defect_qipao.w)
					//+ ", h:" + std::to_string(temp_defect_qipao.h));

					defect_data.emplace_back(temp_defect_qipao);
					defect_data_result.emplace_back(temp_defect_qipao);
				}
				// 从标签映射中提取最大连通域
				//cv::Mat largestCC;
				//labels.copyTo(largestCC);
				//largestCC.setTo(0, labels != maxAreaLabel);
				//largestCC.setTo(255, labels == maxAreaLabel);
				//boundRect = cv::Rect(x + r_tiejiao.x, y + r_tiejiao.y, w, h);


				//defect_data.emplace_back(temp_defect);
				//defect_data_result.emplace_back(temp_defect);

				//continue;
			}
			catch (const std::exception& e)
			{
				spdlog::get("CATL_WCP")->error("传统算法检测打皱气泡失败");
				spdlog::get("CATL_WCP")->error(std::string(e.what()));
				//LogWriterFlush("传统检测未检测到气泡打皱");
				continue;
			}
*/

#pragma endregion

			
		}
	}

	
		

	for (size_t j = 0; j < batch_res.size(); j++) 
	{
		
		float loujinshu_a_o;
		temp_defect.defect_name = classes[(int)batch_res[j].class_id];
		temp_defect.score = batch_res[j].conf;
		temp_defect.defect_id = batch_res[j].class_id;

		cv::Rect r = get_rect(img, batch_res[j].bbox);
		
		if (batch_res[j].conf < m_defect_thresh_map[temp_defect.defect_name])
		{
			//if(m_defect_thresh_map["log_flag"])
			spdlog::get("CATL_WCP")->info(temp_defect.defect_name + "置信度低于阈值 当前推理置信度为" + std::to_string(batch_res[j].conf));
			//LogWriterFlush(temp_defect.defect_name + "置信度低于阈值 当前推理置信度为" + std::to_string(batch_res[j].conf));

			continue;
		}

		//对ok字段的缺陷进行过滤
		if (temp_defect.defect_name == "14_OK_FanGuang" || temp_defect.defect_name == "05_OK_Tiejiao_B" || temp_defect.defect_name == "07_OK_Tiejiao_Y" || temp_defect.defect_name == "18_OK_Tiejiao_W")
		{
			fanguang = true;
			continue;
		}
		else if (temp_defect.defect_name.find("OK") != std::string::npos)
		{
			continue;
		}
		//对贴胶图片的接带和漏金属缺陷进行屏蔽
		/*if (tape_flag)
		{
			if (temp_defect.defect_name == "04_JieDai" || temp_defect.defect_name == "03_HuangBiao"|| temp_defect.defect_name == "10_JiErFanZhe")
				continue;
		}*/
		//阈值卡控，设置参数在ini配置文件中
		//cv::Rect r = get_rect(img, batch_res[j].bbox);
		//对黄标和接带的平均灰度做判断，防止供应商未给贴胶标志位导致的过杀
		else if (temp_defect.defect_name == "10_JiErFanZhe")
		{
			if (r.x > img.cols / 5)
				continue;
		}
		else if (temp_defect.defect_name == "04_JieDai")
		{
			cv::Mat jiedai_huidu_img = img(cv::Rect(r.x, r.y, r.width, r.height));
			float result_mean_jiedai = (cv::mean(jiedai_huidu_img).val[1] + cv::mean(jiedai_huidu_img).val[2]) / 2;
			// 应对厦门白色胶带临时措施 20240708
			if (result_mean_jiedai > 150)
				continue;
			spdlog::get("CATL_WCP")->info("接带平均灰度：" + std::to_string(result_mean_jiedai));
			//LogWriterFlush("接带平均灰度：" + std::to_string(result_mean_jiedai));
			if (r.width< img.cols / 10)
				continue;
			if (result_mean_jiedai < m_defect_thresh_map["huangbiao&jiedai_huidu"])
				continue;
		}
		else if (temp_defect.defect_name == "03_HuangBiao")
		{
			cv::Mat huangbiao_huidu_img = img(cv::Rect(r.x, r.y, r.width, r.height));
			float result_mean_huangbiao = (cv::mean(huangbiao_huidu_img).val[1] + cv::mean(huangbiao_huidu_img).val[2]) / 2;
			spdlog::get("CATL_WCP")->info("黄标平均灰度：" + std::to_string(result_mean_huangbiao));
			//LogWriterFlush("黄标平均灰度：" + std::to_string(result_mean_huangbiao));
			if (r.x< img.cols/6)
				continue;
			if (result_mean_huangbiao < m_defect_thresh_map["huangbiao&jiedai_huidu"])
				continue;
		}
		else if (temp_defect.defect_name == "16_JieDai_2")
		{
			spdlog::get("CATL_WCP")->info("JieDai判断: 置信度" + std::to_string(batch_res[j].conf));
			//LogWriterFlush("JieDai判断: 置信度" + std::to_string(batch_res[j].conf));
			if (batch_res[j].conf < m_defect_thresh_map["04_JieDai"])
			{
				spdlog::get("CATL_WCP")->info(temp_defect.defect_name + "16_JieDai_2置信度低于阈值 当前推理置信度为" + std::to_string(batch_res[j].conf));
				//LogWriterFlush(temp_defect.defect_name + "16_JieDai_2置信度低于阈值 当前推理置信度为" + std::to_string(batch_res[j].conf));
				continue;
			}
			temp_defect.defect_name = "04_JieDai";
			temp_defect.defect_id = 4;
		}
		//else if (temp_defect.defect_name == "11_JiErGenBuKaiLie")
		//{
		//	/*cv::Mat huangbiao_huidu_img = img(cv::Rect(r.x, r.y, r.width, r.height));
		//	float result_mean_huangbiao = cv::mean(huangbiao_huidu_img)[0];
		//	LogWriterFlush("11_JiErGenBuKaiLie平均灰度：" + std::to_string(result_mean_huangbiao));*/
		//	//if (result_mean_huangbiao < 100)
		//	//	continue;
		//	LogWriterFlush("检测到极耳根部开裂，将名称改为极片破损");
		//	temp_defect.defect_name = "01_JiPianPoSun";
		//	temp_defect.defect_id = 1;
		//}
		/*if (batch_res[j].conf < m_defect_thresh_map[temp_defect.defect_name])
		{
			LogWriterFlush(temp_defect.defect_name + "置信度低于阈值 当前推理置信度为" + std::to_string(batch_res[j].conf));
			continue;
		}*/
			
		/*temp_defect.score = batch_res[j].conf;
		temp_defect.defect_id = batch_res[j].class_id;*/
		/*if (temp_defect.defect_name == "16_JieDai_2")
		{
			temp_defect.defect_name = "04_JieDai";
			temp_defect.defect_id = 4;
		}*/
		//极片打皱
		else if (temp_defect.defect_name == "02_JiPianDaZhou")
		{
			if (r.width<20&&r.height<20)
				continue;
			cv::Mat L_img, R_img;
			cv::Point pmx, pmn;
			double mx, mn;
			if (r.x < img.cols / 5 && r.width < img.cols / 5)
			{
				img(cv::Rect(r.x + r.width, r.y, r.width + img.cols / 5, r.height)).copyTo(L_img);
				if (L_img.channels() == 3)
					cv::cvtColor(L_img, L_img, cv::COLOR_BGR2GRAY);
				cv::reduce(L_img, L_img, 0, cv::REDUCE_AVG, CV_64F);
				cv::Sobel(L_img, L_img, CV_64F, 1, 0);
				minMaxLoc(L_img, &mn, &mx, &pmn, &pmx);
				spdlog::get("CATL_WCP")->info("检出缺陷右侧位置灰度为：" + std::to_string(mn));
				//LogWriterFlush("检出缺陷右侧位置灰度为：" + std::to_string(mn));
				if (abs(mn) > 80)
					continue;
			}
			/*LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao.x) + "," + std::to_string(r_tiejiao.y) + " 打皱：" + std::to_string(r.x) + "," + std::to_string(r.y));
			if(r_tiejiao1.width>0)
				LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao1.x) + "," + std::to_string(r_tiejiao1.y) + "  打皱&：" + std::to_string(r.x) + "," + std::to_string(r.y));*/
			/*if (!r_tiejiao.empty())
			{
				if (r.y < r_tiejiao.y + r_tiejiao.height && r.y + r.height>r_tiejiao.y)
					continue;
			}
			if (!r_tiejiao1.empty())
			{
				if (r.y<r_tiejiao1.y + r_tiejiao1.height && r.y + r.height>r_tiejiao1.y)
					continue;
			}*/
			img(cv::Rect(r.x, r.y, r.width, r.height)).copyTo(R_img);
			int contrast;
			int defect_area;
			int mean_val = int(cv::mean(img)[0]);
			calculate_contrast(R_img, 5, defect_area, contrast, mean_val);
			spdlog::get("CATL_WCP")->info(temp_defect.defect_name + "对比度：" + std::to_string(contrast));
			//LogWriterFlush(temp_defect.defect_name + "对比度：" + std::to_string(contrast));
			if (contrast <= m_defect_thresh_map["contrast"])
				continue;
		}
		//极片破损
		else if (temp_defect.defect_name == "01_JiPianPoSun")
		{
			std::cout << r.x << "_" << r.y << "_" << r.width << "_" << r.height << std::endl;
			if (r.width < 20 && r.height < 20)
				continue;
			cv::Mat L_img, R_img;
			cv::Point pmx, pmn;
			double mx, mn;
			//if (r.x < img.cols / 5 && r.width < img.cols / 5)
			//{
			//	// 铝箔版本
			//	continue;

			//	/*img(cv::Rect(r.x + r.width, r.y, r.width + img.cols / 5, r.height)).copyTo(L_img);
			//	if (L_img.channels() == 3)
			//		cv::cvtColor(L_img, L_img, cv::COLOR_BGR2GRAY);
			//	cv::reduce(L_img, L_img, 0, cv::REDUCE_AVG, CV_64F);
			//	cv::Sobel(L_img, L_img, CV_64F, 1, 0);
			//	minMaxLoc(L_img, &mn, &mx, &pmn, &pmx);
			//	LogWriterFlush("检出缺陷右侧位置灰度为：" + std::to_string(mn));
			//	if (abs(mn) > 80)
			//		continue;*/
			//}
			/*LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao.x) + "," + std::to_string(r_tiejiao.y) + " 破损：" + std::to_string(r.x) + "," + std::to_string(r.y));
			if (r_tiejiao1.width > 0)
				LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao1.x) + "," + std::to_string(r_tiejiao1.y) + "  破损：" + std::to_string(r.x) + "," + std::to_string(r.y));*/
			
			/*if (!r_tiejiao.empty())
			{
				if (r.y < r_tiejiao.y + r_tiejiao.height && r.y + r.height>r_tiejiao.y)
					continue;
			}
			if (!r_tiejiao1.empty())
			{
				if (r.y<r_tiejiao1.y + r_tiejiao1.height && r.y + r.height>r_tiejiao1.y)
					continue;
			}*/
			
			img(cv::Rect(r.x, r.y, r.width, r.height)).copyTo(R_img);
			
			int contrast;
			int defect_area;
			int mean_val = int(cv::mean(img)[0]);
			calculate_contrast(R_img, 5, defect_area, contrast, mean_val);
			spdlog::get("CATL_WCP")->info(temp_defect.defect_name + "对比度：" + std::to_string(contrast));
			//LogWriterFlush(temp_defect.defect_name + "对比度：" + std::to_string(contrast));
			/*if (contrast <= m_defect_thresh_map["posun_contrast"])
				continue;*/
		}
		//对漏金属进行长、宽、面积卡控
		else if (temp_defect.defect_name == "12_LouJinShu")
		{
			// LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao.x) + "," + std::to_string(r_tiejiao.y) + " 漏金属：" + std::to_string(r.x) + "," + std::to_string(r.y) + "," + std::to_string(r.width) + "," + std::to_string(r.height));
			/*temp_defect.defect_name = "01_JiPianPoSun";
			temp_defect.defect_id = 1;*/
			/*if (r_tiejiao1.width > 0)
				LogWriterFlush("tiejiaoweizhi：" + std::to_string(r_tiejiao1.x) + "," + std::to_string(r_tiejiao1.y) + " 漏金属：" + std::to_string(r.x) + "," + std::to_string(r.y));*/
			if (!r_tiejiao.empty())
			{
				if (r.y < r_tiejiao.y + r_tiejiao.height && r.y + r.height>r_tiejiao.y)
					continue;
			}
			if (!r_tiejiao1.empty())
			{
				if (r.y<r_tiejiao1.y + r_tiejiao1.height && r.y + r.height>r_tiejiao1.y)
					continue;
			}
			if (r.width > img.cols / 1.2 && r.y <  3)
			{
				spdlog::get("CATL_WCP")->info("LouJinShu：" + std::to_string(r.width) + " > " + std::to_string(img.cols / 1.2));
				//LogWriterFlush("LouJinShu：" + std::to_string(r.width) + " > " + std::to_string(img.cols / 1.2));
				continue;
			}
			bool stop_m = false;
			bool stop_a = false;
			double chang, gao, mianji,m1;
			cv::Mat labels, centroids, stats, res_img;
			res_img = img(cv::Rect(r.x , r.y, r.width, r.height));
			if (res_img.channels() == 3)
			{
				cv::cvtColor(res_img, res_img, cv::COLOR_RGB2GRAY);
			}
			cv::threshold(res_img, res_img, 40, 255, cv::THRESH_BINARY);
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
			cv::morphologyEx(res_img, res_img, cv::MORPH_OPEN, kernel);
			res_img.convertTo(res_img, CV_8UC1);
			int connected_num = cv::connectedComponentsWithStats(res_img, labels, stats, centroids);
			for (int i = 1; i < connected_num; ++i)
			{
				// 宽高不超过设定值像素
				int a = stats.at<int>(i, cv::CC_STAT_WIDTH);

				if ((a >= (m_defect_thresh_map["loujinshu"] / m_defect_thresh_map["calibration"])))
				{
					chang = (stats.at<int>(i, cv::CC_STAT_WIDTH) - 20 )* m_defect_thresh_map["calibration"];
					gao = stats.at<int>(i, cv::CC_STAT_HEIGHT)* m_defect_thresh_map["calibration"];
					stop_m = true;
				}
				if (stats.at<int>(i, cv::CC_STAT_AREA) * 0.66 > (m_defect_thresh_map["loujinshu_area"] / pow(m_defect_thresh_map["calibration"], 2)))
				{
					mianji = (stats.at<int>(i, cv::CC_STAT_AREA) * 0.66) * pow(m_defect_thresh_map["calibration"], 2);
					stop_a = true;
				}
				else
				{
					all_area_loujinshu += stats.at<int>(i, cv::CC_STAT_AREA);
				}
			}
			m1 = m_defect_thresh_map["loujinshu_area"];
			if (all_area_loujinshu * 0.66 > (m_defect_thresh_map["loujinshu_area"] / pow(m_defect_thresh_map["calibration"], 2)))
			{
				mianji = all_area_loujinshu * pow(m_defect_thresh_map["calibration"], 2);
				stop_a = true;
			}
			if (!stop_m || !stop_a)
			{
				continue;
			}
			temp_defect.x = r.x;
			temp_defect.y = r.y;
			temp_defect.h = r.height;
			temp_defect.w = r.width;
			spdlog::get("CATL_WCP")->info("模型推理成功,结果为：" + temp_defect.defect_name + " 阈值为：" + std::to_string(temp_defect.score) + " 宽：" + std::to_string(chang) + " 高：" + std::to_string(gao) + " 面积：" + std::to_string(mianji));
			//LogWriterFlush("模型推理成功,结果为：" + temp_defect.defect_name + " 阈值为：" + std::to_string(temp_defect.score) + " 宽：" + std::to_string(chang) + " 高：" + std::to_string(gao) + " 面积：" + std::to_string(mianji));
			spdlog::get("CATL_WCP")->info("AI process cost:" + std::to_string((double)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count())) + " ms");
			//LogWriterFlush("AI process cost:" + std::to_string((double)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count())) + " ms");
			defect_data.emplace_back(temp_defect);
			defect_data_result.emplace_back(temp_defect);
			result_flag = 1;
			continue;
		}

		//胶纸打皱
		//if (temp_defect.defect_name == "20_JiaoZhiDaZhou")
		//{
		//	// std::cout << "jiaozhi " << std::endl;
		//	if (r.y > boundRect.y && r.y < boundRect.y + boundRect.height)
		//	{
		//		LogWriterFlush("胶纸打皱位置在对贴区");
		//		continue;
		//	}
		//}

		// LY胶纸
		//if (temp_defect.defect_name == "19_JiaoZhiFanZhe" || temp_defect.defect_name == "20_JiaoZhiDaZhou" 
		//	|| temp_defect.defect_name == "21_WeiChongDie")
		//{
		//	if (r.y > r_tiejiao.y + r_tiejiao.height || r.y + r.height < r_tiejiao.y)
		//	{
		//		//LogWriterFlush("胶纸打皱不在胶纸区域");
		//		continue;
		//	}
		//}

		temp_defect.x = r.x;
		temp_defect.y = r.y;
		temp_defect.h = r.height;
		temp_defect.w = r.width;
		spdlog::get("CATL_WCP")->info("模型推理成功,结果为：" + temp_defect.defect_name + " 阈值为：" + std::to_string(temp_defect.score));
		spdlog::get("CATL_WCP")->info("AI process cost:" + std::to_string((double)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count())) + " ms");
		//LogWriterFlush("模型推理成功,结果为：" + temp_defect.defect_name + " 阈值为：" + std::to_string(temp_defect.score));
		//LogWriterFlush("AI process cost:" + std::to_string((double)(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count())) + " ms");
		

		/*if (temp_defect.defect_name == "17_LvBoPoSun" || temp_defect.defect_name == "19_DuanLie" || 
			temp_defect.defect_name == "20_YuLiao" || temp_defect.defect_name == "21_XuHan")
		{
			temp_defect.defect_id = 1;
			temp_defect.defect_name = "01_JiPianPoSun";
		}*/

		defect_data.emplace_back(temp_defect);
		defect_data_result.emplace_back(temp_defect);
		result_flag = 1;
	}
	if (result_flag) {
		spdlog::get("CATL_WCP")->info("模型推理成功,结果为：NG");
		//LogWriterFlush("模型推理成功,结果为：NG");
	}
	/*else {
		LogWriterFlush("模型推理成功,结果为：OK");
	}*/
	if (!tape_flag && defect_data.empty() && !fanguang && temp_defect1.score > m_defect_thresh_map[temp_defect1.defect_name] && temp_defect1.defect_name != "")
	{
		if (!r_tiejiao.empty())
		{
			if (temp_defect1.y < r_tiejiao.y + r_tiejiao.height && temp_defect1.y + temp_defect1.h>r_tiejiao.y)
				return StatusCode::SUCCESS;
		}
		if (!r_tiejiao1.empty())
		{
			if (temp_defect1.y<r_tiejiao1.y + r_tiejiao1.height && temp_defect1.y + temp_defect1.h>r_tiejiao1.y)
				return StatusCode::SUCCESS;
		}
		defect_data.emplace_back(temp_defect1);
		defect_data_result.emplace_back(temp_defect1);
	}

	// HX L50 胶纸气泡检测
	//DetectBubble(srcImg, defect_data, img_x, img_w, y1, y2, m_defect_thresh_map); // 气泡
	DetectDaZhou(img, defect_data, img_x, img_w, XYdataUp, XYdataDown, m_defect_thresh_map, 1); // 打皱


	/*t_cost = (static_cast<double>(cv::getTickCount()) - time1) / cv::getTickFrequency() * 1000;
	std::cout << "after cost(ms): " << t_cost << std::endl;*/
	//cv::rectangle(img, boundRect, cv::Scalar(0, 255, 0), 10);
	return StatusCode::SUCCESS;
}

Detection_J::Detection_J() 
{
	for (size_t i = 0; i < classesname_m.size(); i++)
	{
		m_defect_thresh_map.insert(std::pair<std::string, float>(classesname_m[i], 0.1));
	}
}
Detection_J::~Detection_J()
{
	// Release stream and buffers
	cudaStreamDestroy(stream);
	CUDA_CHECK(cudaFree(buffers[inputIndex]));
	CUDA_CHECK(cudaFree(buffers[outputIndex]));
	// Destroy the engine
	context->destroy();
	engine->destroy();
	runtime->destroy();
}
