#ifndef YOLOV5_D_H
#define YOLOV5_D_H

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <string.h>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include "yololayer.h"
#include <chrono>
#include "cuda_utils.h"
#include "logging.h"
#include "common.hpp"
#include "utils.h"
#include "calibrator.h"
#include "CommonData.h"
#include "m_logger.h"
#include <windows.h>
#include <direct.h>
#include <NvInfer.h>
#include <cuda_runtime.h>
#include <NvInferRuntime.h>
//#define USE_FP16  // 设置推理精度
//#define DEVICE 0  // GPU id
//#define NMS_THRESH 0.1
//#define CONF_THRESH 0.1
//#define BATCH_SIZE 1
//#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 
#define USE_FP16  // set USE_INT8 or USE_FP16 or USE_FP32
#define DEVICE 0  // GPU id
#define NMS_THRESH 0.1
#define CONF_THRESH 0.1
#define BATCH_SIZE 1
#define MAX_IMAGE_INPUT_SIZE_THRESH 3000 * 3000 // ensure it exceed the maximum size in the input images !

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1

//const int g_batch_size = 1;

struct Net_config
{
    float gd; // engine threshold
    float gw;  // engine threshold
    const char* netname;
};


class Detection_J
{
public:
	Detection_J();
	~Detection_J();


	bool IsRuning = false;                                    // 程序是否运行中
	cv::Mat source;                                           // 缩放后的相机图像
	int camType;
	std::map<std::string, float> m_defect_thresh_map;		  // config检测参数
	std::map<std::string, float> m_area_map;
	float calibration;										  // 像素与面积之间的转换比例
	std::vector<DefectData> defect_data_result;               //检测结果
	float loujinshu;
	float area_loujinshu;
	float area_posun;

	////////开放气泡卡控参数//////////
	float qp_c;
	float qp_k;
	float qp_c_calibration;
	float qp_k_calibration;
	

	bool ReadParams(const std::string& file_path);

	//////////2025-04-27 切拉换型联调//////////
	bool Initialize(int camType,const char* model_path, const char* config_path, int num);
	//bool Initialize(int camType, const char* model_path, const char* config_path, int num , int tape_flag);


	bool reload(const char* config_path);
	StatusCode Detecting(cv::Mat& img, std::vector<DefectData>& defect_data,int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, int tape_flag);
	StatusCode DetectingJiaoZhi(cv::Mat& img, std::vector<DefectData>& defect_data, int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, int tape_flag);

	//////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
	StatusCode DetectingGuaiJiao_1(cv::Mat& img, std::vector<DefectData>& defect_data, int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, int tape_flag);

	//////////2025-05-19 LUON01 拐角胶cam1 cam2分开/////////
	StatusCode DetectingGuaiJiao_2(cv::Mat& img, std::vector<DefectData>& defect_data, int img_x, int img_w, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, int tape_flag);

private:
	char netname[20] = { 0 };
	float gd = 0.0f, gw = 0.0f;
	
	//开放气泡打皱卡控参数
	std::vector<std::string> classesname_m{ "01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu","calibration","loujinshu","loujinshu_area","huangbiao&jiedai_huidu","contrast","posun_contrast","qp_c","qp_k","qp_c_calibration","qp_k_calibration","25_GJJ_JiaoZhiDaZhou","26_GJJ_JiaoZhiPoSun" };

	//std::vector<std::string> classesname_m{"01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu","20_JiaoZhiDaZhou", "22_JiaoZhiQiPao","calibration","loujinshu","loujinshu_area","huangbiao&jiedai_huidu","contrast","posun_contrast"};
	

	//////////常规检测类别//////////
	const char* classes[17] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2"};



	//////////JC-mcc检测类别//////////
	//const char* classes[23] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
	//	"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
	//	"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "19_Particle", "20_JiaoZhiDaZhou", "21_DieJiPian", "22_DuoPian", "23_LiePian","24_DuoJiao" };



	/// /////////-------------//////////////////12.11
	//////////2025-07-10 白色拐角胶切断胶检测项合并//////////
	//const char* classes_jiaozhi[2] = { "20_JiaoZhiDaZhou","22_JiaoZhiQiPao" };
	const char* classes_jiaozhi[2] = { "25_LP_JiaoZhiDaZhou","26_LP_JiaoZhiPoSun" };

	//////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
	//////////2025-07-10 白色拐角胶切断胶检测项合并//////////
	//const char* classes_guaijiao[2] = { "25_GJJ_JiaoZhiDaZhou","26_GJJ_JiaoZhiPoSun" };
	const char* classes_guaijiao[2] = { "25_LP_JiaoZhiDaZhou","26_LP_JiaoZhiPoSun" };


	////////////2025-04-09 满足条件时将打皱判定成气泡//////////
	//const int DAZHOU_CLASS_ID = 0;   
	//const int QIPAO_CLASS_ID = 1;    

	// LY 增加胶纸翻折 后续要合并XM余料 序号从19开始
	/*const char* classes[20] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2" ,"19_JiaoZhiFanZhe" , "20_JiaoZhiDaZhou", "21_WeiChongDie" };*/

	//JC L10圆柱电芯检
	/*const char* classes[22] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "19_Particle", "20_JiaoZhiDaZhou", "21_DieJiPian", "22_DuoPian", "23_LiePian" };*/

	//XM 新增余料 方壳皆以这个为准
	/*const char* classes[19] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "17_YuLiao", "18_OK_Tiejiao_W"};*/

	//胶纸气泡。
	/*const char* classes[20] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "18_OK_Tiejiao_W", "20_JiaoZhiDaZhou","22_JiaoZhiQiPao" };*/

	/*const char* classes[20] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "18_OK_Tiejiao_W", "22_JiaoZhiQiPao" };*/

	//cs 铜丝版本 训练的时候单独带上铜丝
	/*const char* classes[18] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "19_Tongsi" };*/

	// 铝箔版本 JC
	/*const char* classes[19] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "17_LvBoPoSun", "18_OK_LvBo"};*/

	// 铝箔版本YB 检测项比JC多
	/*const char* classes[22] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "17_LvBoPoSun", "18_OK_LvBo", "19_DuanLie", "20_YuLiao", "21_XuHan" };*/

	/*****乐色*********/
	//临时加的项 后面不了了之
	/*const char* classes[20] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "20_JiErYiChang", "21_JiErSxing", "22_BoLangBian" };*/

	//临时加的项 后面不了了之
	/*const char* classes[18] = { "00_OK_anban","01_JiPianPoSun","02_JiPianDaZhou","03_HuangBiao","04_JieDai","05_OK_Tiejiao_B",
		"06_OK_ZangWu","07_OK_Tiejiao_Y","08_OK_LenLieWen","09_OK_Hen","10_JiErFanZhe","11_JiErGenBuKaiLie","12_LouJinShu",
		"13_OK_dian","14_OK_FanGuang","15_OK_CuoWei" , "16_JieDai_2", "19_LouAoBan" };*/

	Net_config yolo_nets[4] = {
		{0.33, 0.50, "yolov5s"},
		{0.67, 0.75, "yolov5m"},
		{1.00, 1.00, "yolov5l"},
		{1.33, 1.25, "yolov5x"}
	};

	int CLASS_NUM = 17;
	float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	float prob[BATCH_SIZE * OUTPUT_SIZE];
	size_t size = 0;

	int inputIndex = 0;
	int outputIndex = 0;

	char* trtModelStream = nullptr;
	void* buffers[2] = { 0 };

	nvinfer1::IExecutionContext* context;
	cudaStream_t stream;
	nvinfer1::IRuntime* runtime;
	nvinfer1::ICudaEngine* engine;
};

#endif // 

