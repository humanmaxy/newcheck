#include"pch.h"
#include"JuanraoDetect.h"
#include"GPUinfo.h"
#include"LongCofig.h"
#include"yolov5_d.h"
#include<io.h>
#include<mutex>
#include<future>

//////////2025-05-16 中州拐角胶过杀//////////
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <cstdlib>
#include <thread>
#include <fstream>
#include <filesystem>


using namespace CATL_JUANRAO;
using namespace std;
Detection_J* juanrao = nullptr;
bool juanrao_init = false;
Detection_J* juanrao1 = nullptr;
bool juanrao_init1 = false;

//胶纸打皱、气泡AI检测
Detection_J* juanrao_jiaozhi = nullptr;
bool juanrao_jiaozhi_init = false;
Detection_J* juanrao1_jiaozhi = nullptr;
bool juanrao_jiaozhi_init1 = false;

Detection_J* juanrao2 = nullptr;
bool juanrao_init2 = false;
Detection_J* juanrao3 = nullptr;
bool juanrao_init3 = false;


////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
Detection_J* juanrao_guaijiao = nullptr;
bool juanrao_guaijiao_init = false;
Detection_J* juanrao1_guaijiao = nullptr;
bool juanrao_guaijiao_init1 = false;



int camera_type = 0;

std::string Log_folder_root = "D:/AI_Log/";

//const char* detectModelPath = "e:/users/LuoN01/Desktop/wcp_juanrao/x64/Release/AI_Config/model/wcp_det.engine";// debug
////胶纸打皱、气泡AI检测
//const char* jiaozhidetectModelPath = "e:/users/LuoN01/Desktop/wcp_juanrao/x64/Release/AI_Config/model/wcp_det_jiaozhi.engine";// debug

const char* detectModelPath = "AI_Config/model/wcp_det.engine"; // release
//const char* detectModelPath = "AI_Config/model"; // release
//const char* detectModelPath = "AI_Config/model/wcp_det.engine"; // release

//////////2025-04-27 切拉换型联调//////////
const char* jiaozhidetectModelPath = "AI_Config/model/wcp_det_jiaozhi.engine";// release
//const char* jiaozhidetectModelPath = "AI_Config/model/wcp_det_jiaozhi.wts";// release

//////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
const char* guaijiaodetectModelPath = "AI_Config/model/wcp_det_guaijiao.engine"; // release


//const char* configPath = "e:/users/LuoN01/Desktop/wcp_juanrao/x64/Release/AI_Config/juanrao.ini";// debug
////胶纸打皱、气泡AI检测
//const char* jiaozhiconfigPath = "e:/users/LuoN01/Desktop/wcp_juanrao/x64/Release/AI_Config/juanrao_jiaozhi.ini";// debug
const char* configPath = "AI_Config/juanrao.ini"; // release
const char* jiaozhiconfigPath= "AI_Config/juanrao_jiaozhi.ini"; // release


//////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
const char* guaijiaoconfigPath = "AI_Config/juanrao_guaijiao.ini"; // release


static cv::Mat test_img = cv::Mat::ones(640, 640, CV_8UC3);
std::string ImgSaveDir;
std::string oldTime;
int cropWidthMin = 50000;

//std::string ai_version = "CATL_20241121_TRT8_HXL50_EP";
//std::string ai_version = "CATL_20250321_TRT8_EP_JC-10";
std::string ai_version = "CATL_20250728_FD_LP_v2.1";

//// 获取时间
//std::string getTime()
//{
//    struct tm t_tm;
//    time_t timer;
//    time(&timer);
//    localtime_s(&t_tm, &timer);
//
//    int year = t_tm.tm_year + 1900;   //年
//    int month = t_tm.tm_mon + 1;      //月
//    int day = t_tm.tm_mday;           //日
//
//    std::stringstream  sstr;
//
//    sstr << year;
//    sstr << std::setw(2) << std::setfill('0') << month;
//    sstr << std::setw(2) << std::setfill('0') << day;
//
//    return sstr.str();
//}
//// 存图部分
//inline int creatFile(const std::string& dir)
//{
//    int reuslt = 0;
//
//    if (-1 == _access(dir.c_str(), 0))
//    {
//        reuslt = _mkdir(dir.c_str());
//    }
//    return reuslt;
//}
//
//void InitImgSave()
//{
//    std::string strDir = "D:/";
//    ImgSaveDir = strDir + "AI_Img";
//    creatFile(ImgSaveDir);
//    std::string strDate = getTime();
//    oldTime = strDate;
//    creatFile(ImgSaveDir +"/"+ strDate);
//}
//void ImgSave(cv::Mat& img)
//{
//    time_t rawtime;
//    struct tm* timeinfo;
//    time(&rawtime);
//    timeinfo = localtime(&rawtime);
//    std::string strDate = getTime();
//    std::mutex mtx;
//    mtx.lock();
//    if (strDate != oldTime)
//    {
//        oldTime = strDate;
//        creatFile(ImgSaveDir + "/" + strDate);
//    }
//    mtx.unlock();
//    
//    std::string savePath = ImgSaveDir + "/" + oldTime + "/"
//        + std::to_string(timeinfo->tm_hour) + std::to_string(timeinfo->tm_min) + 
//        std::to_string(timeinfo->tm_sec) + std::to_string(std::rand()) + ".jpg";
//    mtx.lock();
//    cv::imwrite(savePath, img);
//    mtx.unlock();
//    LogWriterInit("保存图片名："+ savePath);
//}

/************************************************************************************************************************
 * Description: 该函数为易鸿加载我们AI程序的API初始化接口函数
 *
 * Input:       @param cameraType    相机型号，由易鸿提供
 *
 * Outpput:     无
 *
 * Return:      函数无返回值
 *
 * Author:      马某
 *
 * Data:        2023.01.08
 *************************************************************************************************************************/
DLL_API StatusCode CATL_JUANRAO::InitModel(int cameraType)
{
    // 防止重复初始化
    if (juanrao_init)
    {
        return StatusCode::CLASS_INSTANCE_ALREADY_EXIST;
    }
    // 检查是否存在NVIDIA GPU
    /*if (NAMESPA::getVideoInfo())
        return StatusCode::NVIDIA_ERROR;*/
    camera_type = cameraType;

    
    //初始化日志

    init_spdlog("CATL_WCP", Log_folder_root + "AI_Log.log", spdlog::level::info);
    spdlog::get("CATL_WCP")->info("\r\n");
    spdlog::get("CATL_WCP")->info("版本" + ai_version);
    
    if (1 == cameraType)
    {

        if (juanrao == nullptr)
        {
            juanrao_init = false;
            juanrao = new Detection_J();
        }
        if (juanrao_jiaozhi == nullptr)
        {
            juanrao_jiaozhi_init = false;
            juanrao_jiaozhi = new Detection_J();
        }

        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
        if (juanrao_guaijiao == nullptr)
        {
            juanrao_guaijiao_init = false;
            juanrao_guaijiao = new Detection_J();
        }


        spdlog::get("CATL_WCP")->info("Cam1 函数【InitModel】初始化相机1");

        //////////2025-04-27 切拉换型联调//////////
        bool flag = juanrao->Initialize(cameraType, detectModelPath, configPath, 0);
        //bool flag = juanrao->Initialize(cameraType, detectModelPath, configPath, 0,0);


        if (!flag)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【InitModel】使用常规模型初始化相机1失败");

            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        //////////2025-04-27 切拉换型联调//////////
        bool flag_jiaozhi = juanrao_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0);
        //bool flag_jiaozhi = juanrao_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0 ,1);

        // 检查校准初始化是否成功
        if (!flag_jiaozhi) {
            spdlog::get("CATL_WCP")->error("Cam1 函数【InitModel】使用切断胶模型初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////

        //////////2025-04-27 切拉换型联调//////////
        bool flag_guaijiao = juanrao_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0);
        //bool flag_guaijiao = juanrao_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0,1);


        if (!flag_guaijiao) {
            spdlog::get("CATL_WCP")->error("Cam1 函数【InitModel】使用拐角胶模型初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        if (flag && flag_jiaozhi && flag_guaijiao) {
            spdlog::get("CATL_WCP")->info("Cam1 函数【InitModel】初始化相机1成功");
        }

        //if (flag && flag_jiaozhi) {
        //    spdlog::get("CATL_WCP")->info("Cam1 函数【InitModel】初始化相机1成功");
        //}

        juanrao_init = true;
        juanrao_jiaozhi_init = true;
    }
    else if (2 == cameraType)
    {   
        juanrao = new Detection_J();
        juanrao_jiaozhi = new Detection_J();
        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
        juanrao_guaijiao = new Detection_J();


        //////////2025-04-27 切拉换型联调//////////
        bool flag1 = juanrao->Initialize(cameraType, detectModelPath, configPath, 0);
        //bool flag1 = juanrao->Initialize(cameraType, detectModelPath, configPath, 0,0);


        if (!flag1)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【InitModel】使用常规模型初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        //////////2025-04-27 切拉换型联调//////////
        bool flag1_jiaozhi = juanrao_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0);
        //bool flag1_jiaozhi = juanrao_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0 ,1);

        // 检查校准初始化是否成功
        if (!flag1_jiaozhi) {
            spdlog::get("CATL_WCP")->error("Cam2 函数【InitModel】使用切断胶模型初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }


        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////

        //////////2025-04-27 切拉换型联调//////////
        bool flag1_guaijiao = juanrao_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0);
        //bool flag1_guaijiao = juanrao_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0,2);

        // 检查校准初始化是否成功
        if (!flag1_guaijiao) {
            spdlog::get("CATL_WCP")->error("Cam2 函数【InitModel】使用拐角胶模型初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        if (flag1 && flag1_jiaozhi && flag1_guaijiao) {
            spdlog::get("CATL_WCP")->info("cam2 函数【initmodel】初始化相机1成功");
        }

        /*if (flag1 && flag1_jiaozhi ) {
            spdlog::get("CATL_WCP")->info("cam1 函数【initmodel】初始化相机1成功");
        }*/


        juanrao1 = new Detection_J();
        juanrao1_jiaozhi = new Detection_J();

        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
        juanrao1_guaijiao = new Detection_J();

        //////////2025-04-27 切拉换型联调//////////
        bool flag2 = juanrao1->Initialize(cameraType, detectModelPath, configPath, 0);
        //bool flag2 = juanrao1->Initialize(cameraType, detectModelPath, configPath, 0,0);

        if (!flag2)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【InitModel】使用常规模型初始化相机2失败");

            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        //////////2025-04-27 切拉换型联调//////////
        bool flag2_jiaozhi = juanrao1_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0);
        //bool flag2_jiaozhi = juanrao1_jiaozhi->Initialize(cameraType, jiaozhidetectModelPath, jiaozhiconfigPath, 0,1);

        // 检查校准初始化是否成功
        if (!flag2_jiaozhi) {
            spdlog::get("CATL_WCP")->error("Cam2 函数【InitModel】使用切断胶模型初始化相机2失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
        // 
        //////////2025-04-27 切拉换型联调//////////
        bool flag2_guaijiao = juanrao1_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0);
        //bool flag2_guaijiao = juanrao1_guaijiao->Initialize(cameraType, guaijiaodetectModelPath, guaijiaoconfigPath, 0 ,2);

        // 检查校准初始化是否成功
        if (!flag2_guaijiao) {
            spdlog::get("CATL_WCP")->error("Cam2 函数【InitModel】使用拐角胶模型初始化相机2失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        if (flag2 && flag2_jiaozhi && flag2_guaijiao) {
            spdlog::get("CATL_WCP")->info("Cam2 函数【InitModel】初始化相机1、2成功");
        }

        /*if (flag2 && flag2_jiaozhi) {
            spdlog::get("CATL_WCP")->info("Cam2 函数【InitModel】初始化相机1、2成功");
        }*/



        // 检测实例初始化成功
        juanrao_init = true;
        juanrao_init1 = true;
        juanrao_jiaozhi_init = true;
        juanrao_jiaozhi_init1 = true;

        ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
        juanrao_guaijiao_init = true;
        juanrao_guaijiao_init1 = true;
    }


    //////////2025-05-30 ZZ拐角胶闪退//////////
    else if (4 == cameraType)
    {
        juanrao = new Detection_J();
        bool flag1 = juanrao->Initialize(cameraType, detectModelPath, configPath, 0);
        if (!flag1)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【InitModel】初始化相机1失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }
        juanrao1 = new Detection_J();
        bool flag2 = juanrao1->Initialize(cameraType, detectModelPath, configPath, 0);
        if (!flag2)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【InitModel】初始化相机2失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }

        juanrao2 = new Detection_J();
        bool flag3 = juanrao2->Initialize(cameraType, detectModelPath, configPath, 0);
        if (!flag3)
        {
            spdlog::get("CATL_WCP")->info("Cam3 函数【InitModel】初始化相机3失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }
        juanrao3 = new Detection_J();
        bool flag4 = juanrao3->Initialize(cameraType, detectModelPath, configPath, 0);
        if (!flag4)
        {
            spdlog::get("CATL_WCP")->info("Cam4 函数【InitModel】初始化相机4失败");
            return StatusCode::INIT_DETECT_HANDLE_FAILED;
        }


        spdlog::get("CATL_WCP")->info("Cam1 Cam2 Cam3 Cam4 函数【InitModel】初始化相机1、2、3、4成功");


        // 检测实例初始化成功
        juanrao_init = true;
        juanrao_init1 = true;
        juanrao_init2 = true;
        juanrao_init3 = true;
    }
    else
    {
        spdlog::get("CATL_WCP")->info("【InitModel】相机参数错误\r\n");
        return StatusCode::INIT_DETECT_HANDLE_FAILED;
    }
    return StatusCode::SUCCESS;
}

DLL_API void CATL_JUANRAO::ReloadParams()
{
    juanrao->reload(configPath);
    juanrao1->reload(configPath);
    //
    juanrao_jiaozhi->reload(jiaozhiconfigPath);
    juanrao1_jiaozhi->reload(jiaozhiconfigPath);

    ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
    juanrao_guaijiao->reload(guaijiaoconfigPath);
    juanrao1_guaijiao->reload(guaijiaoconfigPath);


    
    return;
}

// map转参数结构体
void map2AI_Params(float& parameter, std::string param_name)
{
    if (juanrao->m_defect_thresh_map.count(param_name))
    {
        parameter = juanrao->m_defect_thresh_map[param_name];
    }
    return;
}

void map2AI_Params(int& parameter, std::string param_name)
{
    if (juanrao->m_defect_thresh_map.count(param_name))
    {
        parameter = juanrao->m_defect_thresh_map[param_name];
    }
    return;
}

// 获取AI参数接口
DLL_API AI_Params CATL_JUANRAO::GetParamInfo()
{
    AI_Params pa;
    map2AI_Params(pa.Thr_JiPianPoSun, "01_JiPianPoSun");
    map2AI_Params(pa.Thr_JiPianDaZhou, "02_JiPianDaZhou");
    map2AI_Params(pa.Thr_HuangBiao, "03_HuangBiao");
    map2AI_Params(pa.Thr_JieDai, "04_JieDai");
    map2AI_Params(pa.Thr_JiErFanZhe, "10_JiErFanZhe");
    map2AI_Params(pa.Thr_JiErGenBuKaiLie, "11_JiErGenBuKaiLie");
    map2AI_Params(pa.Thr_Particle, "19_Particle");
    map2AI_Params(pa.Thr_Tongsi, "19_Tongsi");
    map2AI_Params(pa.Thr_LvBoPoSun, "17_LvBoPoSun");

    map2AI_Params(pa.loujinshu_width, "loujinshu");
    map2AI_Params(pa.loujinshu_area, "loujinshu_area");

    map2AI_Params(pa.huangbiao_jiedai_huidu, "huangbiao&jiedai_huidu");
    map2AI_Params(pa.contrast, "contrast");
    map2AI_Params(pa.posun_contrast, "posun_contrast");

    //map2AI_Params(pa.Thr_JiaoZhiDaZhou, "20_JiaoZhiDaZhou");
    //map2AI_Params(pa.Thr_JiaoZhiQiPao, "22_JiaoZhiQiPao");

    ////////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////
    map2AI_Params(pa.Thr_GJJ_JiaoZhiDaZhou, "25_GJJ_JiaoZhiDaZhou");
    map2AI_Params(pa.Thr_GJJ_JiaoZhiQiPao, "26_GJJ_JiaoZhiPoSun");

    return pa;
}

// 获取版本信息
std::string CATL_JUANRAO::GetVervionInfo()
{
    std::string version = ai_version;
    return version;
}

// 获取版本信息
char version[256];
DLL_API char* CATL_JUANRAO::GetVersionInfo()
{
    sprintf(version, ai_version.c_str());
    return version;
}

DLL_API StatusCode CATL_JUANRAO::CatlDetect(const cv::Mat& source, int x, int y, int w, int h, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, std::vector<DefectData>& defect_data, int tape_flag)
{

    spdlog::get("CATL_WCP")->info("Cam1 检测算法开始");
    double t1 = static_cast<double>(cv::getTickCount());// 检测开始时间

    spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的x, y, w, h为：" + std::to_string(x) + std::string(",") + std::to_string(y) + std::string(",") + std::to_string(w) + std::string(",") + std::to_string(h));
    defect_data.clear();
    if (source.empty())
    {
        spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的图像为空");
        return StatusCode::ROI_IMG_EMPTY;
    }

    //////////2025-02-08//////////

    if (tape_flag == 0) {

        spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的tape_flage=0");
    
        if (w <= 0 || h <= 0)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的ROI信息有误");
            return StatusCode::ROI_IMG_EMPTY;
        }
        cv::Mat temp1 = source.clone();
        cv::Mat temp2 = source.clone();

        int LOG_FLAG = 0;
        LOG_FLAG = juanrao->m_defect_thresh_map["log_flag"];
        if (LOG_FLAG)
            spdlog::get("CATL_WCP")->info("Cam1 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
     /*cv::Mat saveTest = source.clone();
     ImgSave(saveTest);*/
        try {
            source(cv::Rect(x, y, w , h)).copyTo(temp1);
            if (temp1.channels() == 1)
                cv::cvtColor(temp1, temp1, cv::COLOR_GRAY2BGR);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的x,y,w,h信息错误");
            spdlog::get("CATL_WCP")->info("Cam1 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
            spdlog::get("CATL_WCP")->info("Cam1 " + std::string(e.what()));
            return StatusCode::ROI_IMG_EMPTY;
         }

        if (!juanrao->IsRuning)
        {
            juanrao->IsRuning = true;
        }
        if (!juanrao_jiaozhi->IsRuning)
        {
            juanrao_jiaozhi->IsRuning = true;
        }
        //cv::Mat saveImg = temp1.clone();
        cv::Mat img1, img2;

        if (w > cropWidthMin)
        {
            temp1(cv::Rect(0, 0, w / 2, h)).copyTo(img1);
            temp1(cv::Rect(w / 2, 0, w / 2 - 1, h)).copyTo(img2);
            temp1 = img1;
        }
        StatusCode ret;

        try
        {   
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】常规开始检测");
            ret = juanrao->Detecting(temp1, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】常规检测异常");
            spdlog::get("CATL_WCP")->info("Cam1 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;
        }
        /////////2025-03-14解决JC坐标框偏移//////////
        for (size_t j = 0; j < defect_data.size(); ++j)
        {
            defect_data[j].x += x;
            defect_data[j].y += y;
        }
        spdlog::get("CATL_WCP")->info("Cam1常规检测缺陷数量: " + std::to_string(defect_data.size()));
        return ret;

        if (w > cropWidthMin)
        {
            std::vector<DefectData> tempDefect;
            try
            {
                ///////////2025-4-23 LUON01加日志分析闪退问题//////////
                spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】常规检测开始");
                ret = juanrao->Detecting(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
            }
            catch (const std::exception& e)
            {
                spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】常规检测异常");
                spdlog::get("CATL_WCP")->info("Cam1 " + std::string(e.what()));
                return StatusCode::DETECT_ERROR;
            }
            int imgFlag = false;
            for (size_t j = 0; j < tempDefect.size(); ++j)
            {
                tempDefect[j].x += x;
                tempDefect[j].x += w / 2;
                tempDefect[j].y += y;
                defect_data.emplace_back(tempDefect[j]);
            }
        }
    }


    //胶纸打皱、气泡AI检测
    if (tape_flag == 1) {

        spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的tape_flage=1");

        if (XYdataUp.size() < 4)
        {
            spdlog::get("CATL_WCP")->info("Cam1 切断胶传入XYdataUp点个数为" + std::to_string(XYdataUp.size()) + "，应为4");
            return StatusCode::TAPE_POSITION_ERR;
        }
        if (XYdataDown.size() < 4)
        {
            spdlog::get("CATL_WCP")->info("Cam1 切断胶传入XYdataDown点个数为" + std::to_string(XYdataDown.size()) + "，应为4");
            return StatusCode::TAPE_POSITION_ERR;
        }
        for (int i = 0; i < XYdataUp.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入切断胶的x y为：" + std::to_string(XYdataUp[i].first) + std::string(",") + std::to_string(XYdataUp[i].second));
        }
        for (int i = 0; i < XYdataDown.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入切断胶的x y为：" + std::to_string(XYdataDown[i].first) + std::string(",") + std::to_string(XYdataDown[i].second));
        }


        StatusCode ret;
        cv::Mat temp2 = source.clone();

        try
        {
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】切断胶检测开始");
            ret = juanrao_jiaozhi->DetectingJiaoZhi(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】切断胶检测异常");
            spdlog::get("CATL_WCP")->info("Cam1 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;
        }

        spdlog::get("CATL_WCP")->info("Cam1切断胶检测缺陷数量: " + std::to_string(defect_data.size()));
        return ret;
    }


        //////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////

    if (tape_flag == 2) {

        spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入的tape_flage=2");

        if (XYdataUp.size() < 2)
        {
            spdlog::get("CATL_WCP")->info("Cam1 拐角胶传入XYdataUp点个数为" + std::to_string(XYdataUp.size()) + "，应为2");
            return StatusCode::TAPE_POSITION_ERR;
        }
        if (XYdataDown.size() < 2)
        {
            spdlog::get("CATL_WCP")->info("Cam1 拐角胶传入XYdataDown点个数为" + std::to_string(XYdataDown.size()) + "，应为2");
            return StatusCode::TAPE_POSITION_ERR;
        }
        for (int i = 0; i < XYdataUp.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入拐角胶的x y为：" + std::to_string(XYdataUp[i].first) + std::string(",") + std::to_string(XYdataUp[i].second));
        }
        for (int i = 0; i < XYdataDown.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】传入拐角胶的x y为：" + std::to_string(XYdataDown[i].first) + std::string(",") + std::to_string(XYdataDown[i].second));
        }
        StatusCode ret;
        cv::Mat temp2 = source.clone();
        try
        {
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】拐角胶检测开始");

            ////////////2025-05-16 中州拐角胶过杀//////////
            //namespace fs1 = std::filesystem;
            //std::string dir_path1 = "D:/AI_Log/CAM1";
            //if (!fs1::exists(dir_path1)) {
            //    fs1::create_directories(dir_path1);
            //}
            //std::string file_name1 = dir_path1 + "/" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp";
            //cv::imwrite(file_name1, temp2);
            //spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】已保存拐角胶图片");


            ret = juanrao_guaijiao->DetectingGuaiJiao_1(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam1 函数【CatlDetect】拐角胶检测异常");
            spdlog::get("CATL_WCP")->info("Cam1 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;
        }

        spdlog::get("CATL_WCP")->info("Cam1拐角胶检测缺陷数量: " + std::to_string(defect_data.size()));

        double t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        spdlog::get("CATL_WCP")->info("Cam1 检测耗时：" + std::to_string(t_cost) + "ms");




        //////////2025-05-16 中州拐角胶过杀//////////
        //namespace fs2 = std::filesystem;
        //std::string dir_path2 = "D:/AI_Log/CAM1_res";
        //if (!fs2::exists(dir_path2)) {
        //    fs2::create_directories(dir_path2);
        //}
        //for (size_t i = 0; i < defect_data.size(); ++i) {
        //    std::cout << "缺陷" << i << ":" << defect_data[i].defect_name << std::endl;

        //    // 绘制矩形框
        //    cv::rectangle(temp2, cv::Rect(defect_data[i].x, defect_data[i].y, defect_data[i].w, defect_data[i].h), cv::Scalar(25, 255, 255), 1);

        //    // 添加文本标签
        //    cv::putText(temp2, defect_data[i].defect_name + std::to_string(defect_data[i].score), cv::Point(defect_data[i].x, defect_data[i].y + 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 25, 255), 2);

        //    // 检查是否需要保存图片
        //    if (i < 10) {
        //        std::string file_name2 = dir_path2 + "/image_" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp";
        //        cv::imwrite(file_name2, temp2);
        //    }

        //}




        return ret;
    }

}


DLL_API StatusCode CATL_JUANRAO::CatlDetectCam2(const cv::Mat& source, int x, int y, int w, int h, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, std::vector<DefectData>& defect_data, int tape_flag)
{
    
    spdlog::get("CATL_WCP")->info("Cam2 检测算法开始");
    double t2 = static_cast<double>(cv::getTickCount());// 检测开始时间
    
    if (camera_type != 2)
    {
        spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入camera_type不为2");
        return StatusCode::CAMERA_TYPE_ERR;
    }

    spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的x, y, w, h为：" + std::to_string(x) + std::string(",") + std::to_string(y) + std::string(",") + std::to_string(w) + std::string(",") + std::to_string(h));

    defect_data.clear();
    if (source.empty())
    {
        spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的图像为空");
        return StatusCode::ROI_IMG_EMPTY;
    }

    if(tape_flag == 0){

        spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的tape_flage=0");

    // 校验ROI宽度和高度信息
        if (w <= 0 || h <= 0)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的ROI信息有误");
            return StatusCode::ROI_IMG_EMPTY;
        }
        cv::Mat temp1 = source.clone();
        cv::Mat temp2 = source.clone();
        int LOG_FLAG = 0;
        LOG_FLAG = juanrao1->m_defect_thresh_map["log_flag"];
        if (LOG_FLAG)
            spdlog::get("CATL_WCP")->info("Cam2 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
        /*cv::Mat saveTest = source.clone();
        ImgSave(saveTest);*/

        try {
            source(cv::Rect(x, y, w, h)).copyTo(temp1);
            if (temp1.channels() == 1)
                cv::cvtColor(temp1, temp1, cv::COLOR_GRAY2BGR);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的x,y,w,h信息错误");
            spdlog::get("CATL_WCP")->info("Cam2 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
            spdlog::get("CATL_WCP")->info("Cam2 " + std::string(e.what()));
            return StatusCode::ROI_IMG_EMPTY;
        }

        if (!juanrao1->IsRuning)
        {
            juanrao1->IsRuning = true;
        }
        if (!juanrao1_jiaozhi->IsRuning)
        {
            juanrao1_jiaozhi->IsRuning = true;
        }
        //cv::Mat saveImg = temp1.clone();
        cv::Mat img1, img2;
        if (w > cropWidthMin)
        {
            temp1(cv::Rect(0, 0, w / 2, h)).copyTo(img1);
            temp1(cv::Rect(w / 2, 0, w / 2-1, h)).copyTo(img2);
            temp1 = img1;
        }
        StatusCode ret;
        try
        {
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】常规检测开始");
            ret = juanrao1->Detecting(temp1, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】常规检测异常");
            spdlog::get("CATL_WCP")->info("Cam2 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;
        }

        for (size_t j = 0; j < defect_data.size(); ++j)
        {
            defect_data[j].x += x;
            defect_data[j].y += y;
        }
        spdlog::get("CATL_WCP")->info("Cam2常规检测缺陷数量: " + std::to_string(defect_data.size()));
        return ret;

        if (w > cropWidthMin)
        {
            std::vector<DefectData> tempDefect;
            try
            {
                ///////////2025-4-23 LUON01加日志分析闪退问题//////////
                spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】常规检测开始");
                ret = juanrao1->Detecting(img2, tempDefect, x, w, XYdataUp, XYdataDown, tape_flag);
            }
            catch (const std::exception& e)
            {
                spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】常规检测异常");
                spdlog::get("CATL_WCP")->info("Cam2 " + std::string(e.what()));
                return StatusCode::DETECT_ERROR;
            }
            for (size_t j = 0; j < tempDefect.size(); ++j)
            {
                tempDefect[j].x += x;
                tempDefect[j].x += w / 2;
                tempDefect[j].y += y;
                defect_data.emplace_back(tempDefect[j]);
            }
        }
    }

    //胶纸打皱、气泡AI检测
    if (tape_flag == 1) {

        spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的tape_flage=1");

        if (XYdataUp.size() < 4)
        {
            spdlog::get("CATL_WCP")->info("Cam2 传入XYdataUp点个数为" + std::to_string(XYdataUp.size()) + "，应为4");
            return StatusCode::TAPE_POSITION_ERR;
        }
        if (XYdataDown.size() < 4)
        {
            spdlog::get("CATL_WCP")->info("Cam2 传入XYdataDown点个数为" + std::to_string(XYdataDown.size()) + "，应为4");
            return StatusCode::TAPE_POSITION_ERR;
        }
        for (int i = 0; i < XYdataUp.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的x y为：" + std::to_string(XYdataUp[i].first) + std::string(",") + std::to_string(XYdataUp[i].second));
        }
        for (int i = 0; i < XYdataDown.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的x y为：" + std::to_string(XYdataDown[i].first) + std::string(",") + std::to_string(XYdataDown[i].second));
        }


        StatusCode ret;
        cv::Mat temp2 = source.clone();

        try
        {
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】切断胶检测开始");
            ret = juanrao1_jiaozhi->DetectingJiaoZhi(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);

        }

        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】切断胶检测异常");
            spdlog::get("CATL_WCP")->info("Cam2 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;

        }

        spdlog::get("CATL_WCP")->info("Cam2切断胶检测缺陷数量: " + std::to_string(defect_data.size()));
        return ret;
    }


    //////////20225-04-10 LUON01 拐角胶打皱、破损检测//////////

    if (tape_flag == 2) {

        spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入的tape_flage=2");

        if (XYdataUp.size() < 2)
        {
            spdlog::get("CATL_WCP")->info("Cam2 拐角胶传入XYdataUp点个数为" + std::to_string(XYdataUp.size()) + "，应为2");
            return StatusCode::TAPE_POSITION_ERR;
        }
        if (XYdataDown.size() < 2)
        {
            spdlog::get("CATL_WCP")->info("Cam2 拐角胶传入XYdataDown点个数为" + std::to_string(XYdataDown.size()) + "，应为2");
            return StatusCode::TAPE_POSITION_ERR;
        }
        for (int i = 0; i < XYdataUp.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入拐角胶的x y为：" + std::to_string(XYdataUp[i].first) + std::string(",") + std::to_string(XYdataUp[i].second));
        }
        for (int i = 0; i < XYdataDown.size(); i++)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】传入拐角胶的x y为：" + std::to_string(XYdataDown[i].first) + std::string(",") + std::to_string(XYdataDown[i].second));
        }
        StatusCode ret;
        cv::Mat temp2 = source.clone();
        try
        {
            ///////////2025-4-23 LUON01加日志分析闪退问题//////////
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】拐角胶检测开始");

            ////////////2025-05-16 中州拐角胶过杀//////////
            //namespace fs3 = std::filesystem;
            //std::string dir_path3 = "D:/AI_Log/CAM2";
            //if (!fs3::exists(dir_path3)) {
            //    fs3::create_directories(dir_path3);
            //}
            //std::string file_name3 = dir_path3 + "/" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp";
            //cv::imwrite(file_name3, temp2);
            //spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】已保存拐角胶图片");

            //////////2025-05-30 中州拐角胶检测异常及闪退//////////
            //ret = juanrao_guaijiao->DetectingGuaiJiao_2(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
            ret = juanrao1_guaijiao->DetectingGuaiJiao_2(temp2, defect_data, x, w, XYdataUp, XYdataDown, tape_flag);
        }
        catch (const std::exception& e)
        {
            spdlog::get("CATL_WCP")->info("Cam2 函数【CatlDetectCam2】拐角胶检测异常");
            spdlog::get("CATL_WCP")->info("Cam2 " + std::string(e.what()));
            return StatusCode::DETECT_ERROR;
        }

        spdlog::get("CATL_WCP")->info("Cam2拐角胶检测缺陷数量: " + std::to_string(defect_data.size()));


        double t_cost = (static_cast<double>(cv::getTickCount()) - t2) / cv::getTickFrequency() * 1000;
        spdlog::get("CATL_WCP")->info("Cam2 检测耗时：" + std::to_string(t_cost) + "ms");


        ////////////2025-05-16 中州拐角胶过杀//////////
        //namespace fs4 = std::filesystem;
        //std::string dir_path4 = "D:/AI_Log/CAM2_res";
        //if (!fs4::exists(dir_path4)) {
        //    fs4::create_directories(dir_path4);
        //}
        //for (size_t i = 0; i < defect_data.size(); ++i) {
        //    std::cout << "缺陷" << i << ":" << defect_data[i].defect_name << std::endl;

        //    // 绘制矩形框
        //    cv::rectangle(temp2, cv::Rect(defect_data[i].x, defect_data[i].y, defect_data[i].w, defect_data[i].h), cv::Scalar(25, 255, 255), 1);

        //    // 添加文本标签
        //    cv::putText(temp2, defect_data[i].defect_name + std::to_string(defect_data[i].score), cv::Point(defect_data[i].x, defect_data[i].y + 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 25, 255), 2);

        //    // 检查是否需要保存图片
        //    if (i < 10) {
        //        std::string file_name4 = dir_path4 + "/image_" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp";
        //        cv::imwrite(file_name4, temp2);
        //    }

        //}



        return ret;
    }

}

//DLL_API StatusCode CATL_JUANRAO::CatlDetectCam3(const cv::Mat& source, int x, int y, int w, int h, std::vector<DefectData>& defect_data, int tape_flag)
//{
//    //double t1;  // 检测时间记录
//    //double t_cost;  // 检测时间统计
//    //t1 = static_cast<double>(cv::getTickCount());
//    defect_data.clear();
//    if (source.empty())
//    {
//        LogWriterInit("函数【CatlDetectCam3】传入的图像为空。");
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//
//    // 校验ROI宽度和高度信息
//    if (w <= 0 || h <= 0)
//    {
//        LogWriterInit("函数【CatlDetectCam3】传入的ROI信息有误。");
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//    cv::Mat temp1;
//
//    int LOG_FLAG = 0;
//    LOG_FLAG = juanrao2->m_defect_thresh_map["log_flag"];
//    if (LOG_FLAG)
//        LogWriterFlush("cam3 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
//    /*cv::Mat saveTest = source.clone();
//    ImgSave(saveTest);*/
//    try {
//        source(cv::Rect(x, y, w, h)).copyTo(temp1);
//        if (temp1.channels() == 1)
//            cv::cvtColor(temp1, temp1, cv::COLOR_GRAY2BGR);
//    }
//    catch (const std::exception& e)
//    {
//        LogWriterInit("函数【CatlDetectCam3】传入的x,y,w,h信息错误。");
//        LogWriterFlush("cam3 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
//        LogWriterInit(e.what());
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//    //cv::imwrite("D:/AI/" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp", temp1);
//    if (!juanrao2->IsRuning)
//    {
//        juanrao2->IsRuning = true;
//    }
//    //cv::Mat saveImg = temp1.clone();
//    cv::Mat img1, img2;
//
//    if (w > cropWidthMin)
//    {
//        temp1(cv::Rect(0, 0, w / 2, h)).copyTo(img1);
//        temp1(cv::Rect(w / 2, 0, w / 2 - 1, h)).copyTo(img2);
//        temp1 = img1;
//    }
//    StatusCode ret;
//
//    try
//    {
//        ret = juanrao2->Detecting(temp1, defect_data, tape_flag);
//    }
//    catch (const std::exception& e)
//    {
//        LogWriterInit("函数【CatlDetect】检测异常。");
//        LogWriterInit(e.what());
//        /*try
//        {
//            ImgSave(saveImg);
//        }
//        catch (const std::exception&)
//        {
//            LogWriterInit("函数【CatlDetect】存图异常。");
//            return StatusCode::ROI_IMG_EMPTY;
//        }*/
//        return StatusCode::SUCCESS;
//    }
//    int imgFlag = false;
//    for (size_t j = 0; j < defect_data.size(); ++j)
//    {
//        defect_data[j].x += x;
//        defect_data[j].y += y;
//    }
//    if (w > cropWidthMin)
//    {
//        std::vector<DefectData> tempDefect;
//        try
//        {
//            ret = juanrao2->Detecting(img2, tempDefect, tape_flag);
//        }
//        catch (const std::exception& e)
//        {
//            LogWriterInit("函数【CatlDetect】检测异常。");
//            LogWriterInit(e.what());
//            return StatusCode::SUCCESS;
//        }
//        int imgFlag = false;
//        for (size_t j = 0; j < tempDefect.size(); ++j)
//        {
//            tempDefect[j].x += x;
//            tempDefect[j].x += w / 2;
//            tempDefect[j].y += y;
//            defect_data.emplace_back(tempDefect[j]);
//        }
//    }
//
//    /*t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
//    LogWriterInit("cam1检测耗时" + std::to_string(t_cost) + "ms");*/
//    //std::cout << "A1 processing cost(ms): " << (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000 << std::endl;
//    return ret;
//}


//DLL_API StatusCode CATL_JUANRAO::CatlDetectCam4(const cv::Mat& source, int x, int y, int w, int h, std::vector<DefectData>& defect_data, int tape_flag)
//{
//    //double t1;  // 检测时间记录
//    //double t_cost;  // 检测时间统计
//    //t1 = static_cast<double>(cv::getTickCount());
//    if (camera_type != 2)
//    {
//        LogWriterInit("函数【CatlDetectCam4】传入camera_type不为2。");
//        return StatusCode::CAMERA_TYPE_ERR;
//    }
//    defect_data.clear();
//    if (source.empty())
//    {
//        LogWriterInit("函数【CatlDetectCam4】传入的图像为空。");
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//
//    // 校验ROI宽度和高度信息
//    if (w <= 0 || h <= 0)
//    {
//        LogWriterInit("函数【CatlDetectCam4】传入的ROI信息有误。");
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//    cv::Mat temp1;
//    int LOG_FLAG = 0;
//    LOG_FLAG = juanrao3->m_defect_thresh_map["log_flag"];
//    if (LOG_FLAG)
//        LogWriterFlush("cam4 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
//    /*cv::Mat saveTest = source.clone();
//    ImgSave(saveTest);*/
//    try {
//        source(cv::Rect(x, y, w, h)).copyTo(temp1);
//        if (temp1.channels() == 1)
//            cv::cvtColor(temp1, temp1, cv::COLOR_GRAY2BGR);
//    }
//    catch (const std::exception& e)
//    {
//        LogWriterInit("函数【CatlDetectCam4】传入的x,y,w,h信息错误。");
//        LogWriterFlush("cam4 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
//        LogWriterInit(e.what());
//        return StatusCode::ROI_IMG_EMPTY;
//    }
//    //cv::imwrite("D:/AI/" + std::to_string(static_cast<int>(cv::getTickCount())) + ".bmp", temp1);
//    if (!juanrao3->IsRuning)
//    {
//        juanrao3->IsRuning = true;
//    }
//    //cv::Mat saveImg = temp1.clone();
//    cv::Mat img1, img2;
//    if (w > cropWidthMin)
//    {
//        temp1(cv::Rect(0, 0, w / 2, h)).copyTo(img1);
//        temp1(cv::Rect(w / 2, 0, w / 2 - 1, h)).copyTo(img2);
//        temp1 = img1;
//    }
//    StatusCode ret;
//    try
//    {
//        ret = juanrao3->Detecting(temp1, defect_data, tape_flag);
//    }
//    catch (const std::exception& e)
//    {
//        LogWriterInit("函数【CatlDetectCam4】检测异常。");
//        LogWriterInit(e.what());
//        /*try
//        {
//            ImgSave(saveImg);
//        }
//        catch (const std::exception&)
//        {
//            LogWriterInit("函数【CatlDetectCam2】存图异常。");
//            return StatusCode::ROI_IMG_EMPTY;
//        }*/
//        return StatusCode::SUCCESS;
//    }
//    //int num1 = defect_data.size();
//    for (size_t j = 0; j < defect_data.size(); ++j)
//    {
//        defect_data[j].x += x;
//        defect_data[j].y += y;
//    }
//
//    if (w > cropWidthMin)
//    {
//        std::vector<DefectData> tempDefect;
//        try
//        {
//            ret = juanrao3->Detecting(img2, tempDefect, tape_flag);
//        }
//        catch (const std::exception& e)
//        {
//            LogWriterInit("函数【CatlDetectCam4】检测异常。");
//            LogWriterInit(e.what());
//            return StatusCode::SUCCESS;
//        }
//        for (size_t j = 0; j < tempDefect.size(); ++j)
//        {
//            tempDefect[j].x += x;
//            tempDefect[j].x += w / 2;
//            tempDefect[j].y += y;
//            defect_data.emplace_back(tempDefect[j]);
//        }
//    }
//    /*t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
//    LogWriterInit("cam2检测耗时" + std::to_string(t_cost) + "ms");*/
//    //std::cout << "A2 processing cost(ms): " << (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000 << std::endl;
//    return ret;
//}


// C#版本AI调用接口

// 全局存放缺陷数据变量
std::vector<DefectData> g_defect_data, g_defect_data_cam2;
std::vector<cv::Mat> g_img_vec(3), g_img_cam2_vec(3);

DLL_API  StatusCode CATL_JUANRAO::CatlDetectCSharp(unsigned char* r_img_ptr, unsigned char* g_img_ptr, unsigned char* b_img_ptr, int src_img_width, \
    int src_img_height, int x, int y, int w, int h, DefectDataCSharp* defect_data_ptr, int& defect_num, TapePosition* tape_pos)
{
    g_defect_data.clear();
    g_img_vec.clear();
    cv::Mat temp;
    try
    {
        g_img_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), b_img_ptr));
        g_img_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), g_img_ptr));
        g_img_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), r_img_ptr));
        cv::merge(g_img_vec, temp);
    }
    catch (const std::exception&)
    {
        //LogWriterInit("函数【MainFunction】传入的图像异常。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    if (temp.empty())
    {
        //LogWriterInit("函数【MainFunction】传入的图像为空。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    if (w <= 0 || h <= 0)
    {
        //LogWriterInit("函数【MainFunction】传入的ROI信息有误。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    int LOG_FLAG = 0;
    LOG_FLAG = juanrao->m_defect_thresh_map["log_flag"];
    if (LOG_FLAG)
        //LogWriterFlush("cam1 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));
    
    

    try {
        if (w > 15)
            w -= 15;
        temp(cv::Rect(x, y, w, h)).copyTo(temp);
        if (temp.channels() == 1)
            cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);

        //for (int tape_index = 0; tape_index < 4; ++tape_index)
        //{
        //    if (tape_index > 1)
        //        break;
        //    // 参数检验
        //    if (tape_pos[tape_index].top_y < 0)
        //        continue;
        //    // 底部坐标小于0，不继续
        //    if (tape_pos[tape_index].bottom_y < 0)
        //        continue;
        //    // 贴胶顶部坐标大于图像高度，不继续
        //    if (tape_pos[tape_index].top_y > temp.rows)
        //        continue;
        //    if (tape_pos[tape_index].bottom_y > temp.rows)
        //        tape_pos[tape_index].bottom_y = temp.rows;

        //    LogWriterInit("Tape pos cam1：" + std::to_string(tape_pos[tape_index].bottom_y) + std::to_string(tape_pos[tape_index].top_y));
        //    cv::rectangle(temp, cv::Point(0, tape_pos[tape_index].top_y), cv::Point(temp.cols, tape_pos[tape_index].bottom_y), cv::Scalar(0, 0, 0), -1);
        //}
    }
    catch (const std::exception& e)
    {
        //LogWriterInit("函数【MainFunction】传入的x,y,w,h信息错误。");
        //LogWriterInit(e.what());
        return StatusCode::ROI_IMG_EMPTY;
    }
    //createLogFolder("");

    if (!juanrao->IsRuning)
    {
        juanrao->IsRuning = true;
    }
    StatusCode ret = StatusCode::SUCCESS;
    try
    {
       // ret = juanrao->Detecting(temp, g_defect_data, 0, 0, 0, 0, 0);
    }
    catch (const std::exception& e)
    {
        //LogWriterInit("函数【MainFunction】检测异常。");
        //LogWriterInit(e.what());
        /*try
        {
            ImgSave(temp);
        }
        catch (const std::exception&)
        {
            LogWriterInit("函数【MainFunction】存图异常。");
            return StatusCode::ROI_IMG_EMPTY;
        }*/
        return StatusCode::SUCCESS;
    }
    
    // 将缺陷位置信息偏移到原图上
    defect_num = std::min(20, int(g_defect_data.size()));
    try
    {
        // DefectDataCSharp* defect_arr;
        // 构建C#返回值
        for (size_t j = 0; j < defect_num; ++j)
        {
            // g_defect_data[j].y += y;
            defect_data_ptr[j].defect_id = g_defect_data[j].defect_id;
            std::strcpy(defect_data_ptr[j].defect_name, g_defect_data[j].defect_name.c_str());
            defect_data_ptr[j].x = g_defect_data[j].x + x;
            defect_data_ptr[j].y = g_defect_data[j].y + y;
            defect_data_ptr[j].w = g_defect_data[j].w;
            defect_data_ptr[j].h = g_defect_data[j].h;
            defect_data_ptr[j].score = g_defect_data[j].score;
        }
    }
    catch (const std::exception& e)
    {
        //LogWriterInit(e.what());
        return StatusCode::CSHARPFAIL;
    }

    return ret;
}

DLL_API  StatusCode CATL_JUANRAO::CatlDetectCam2CSharp(unsigned char* r_img_ptr, unsigned char* g_img_ptr, unsigned char* b_img_ptr, int src_img_width, \
    int src_img_height, int x, int y, int w, int h, DefectDataCSharp* defect_data_ptr, int& defect_num, TapePosition* tape_pos)
{
    g_defect_data_cam2.clear();
    g_img_cam2_vec.clear();
    
    cv::Mat temp;
    try
    {
        g_img_cam2_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), b_img_ptr));
        g_img_cam2_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), g_img_ptr));
        g_img_cam2_vec.emplace_back(cv::Mat(src_img_height, src_img_width, CV_MAKETYPE(CV_8U, 1), r_img_ptr));
        cv::merge(g_img_cam2_vec, temp);
    }
    catch (const std::exception&)
    {
        //LogWriterInit("函数【MainFunction2】传入的图像异常。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    
    if (temp.empty())
    {
        //LogWriterInit("函数【MainFunction2】传入的图像为空。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    // 校验ROI宽度和高度信息
    if (w <= 0 || h <= 0)
    {
        //LogWriterInit("函数【MainFunction2】传入的ROI信息有误。");
        return StatusCode::ROI_IMG_EMPTY;
    }
    int LOG_FLAG = 0;
    LOG_FLAG = juanrao1->m_defect_thresh_map["log_flag"];
    if (LOG_FLAG)
        //LogWriterFlush("cam2 roi info :" + std::to_string(x) + "," + std::to_string(y) + "," + std::to_string(w) + "," + std::to_string(h));

    try {
        if (w > 15)
            w -= 15;
        temp(cv::Rect(x, y, w, h)).copyTo(temp);
        if (temp.channels() == 1)
            cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);

        //for (int tape_index = 0; tape_index < 4; ++tape_index)
        //{
        //    if (tape_index > 1)
        //        break;
        //    // 参数检验
        //    if (tape_pos[tape_index].top_y < 0)
        //        continue;
        //    // 底部坐标小于0，不继续
        //    if (tape_pos[tape_index].bottom_y < 0)
        //        continue;
        //    // 贴胶顶部坐标大于图像高度，不继续
        //    if (tape_pos[tape_index].top_y > temp.rows)
        //        continue;
        //    if (tape_pos[tape_index].bottom_y > temp.rows)
        //        tape_pos[tape_index].bottom_y = temp.rows;

        //    LogWriterInit("Tape pos cam2：" + std::to_string(tape_pos[tape_index].bottom_y) + std::to_string(tape_pos[tape_index].top_y));
        //    cv::rectangle(temp, cv::Point(0, tape_pos[tape_index].top_y), cv::Point(temp.cols, tape_pos[tape_index].bottom_y), cv::Scalar(0, 0, 0), -1);
        //}
    }
    catch (const std::exception& e)
    {
        //LogWriterInit("函数【MainFunction2】传入的x,y,w,h信息错误。");
        //LogWriterInit(e.what());
        return StatusCode::ROI_IMG_EMPTY;
    }
    //createLogFolder("");

    if (!juanrao1->IsRuning)
    {
        juanrao1->IsRuning = true;
    }
    StatusCode ret = StatusCode::SUCCESS;
    try
    {
       // ret = juanrao1->Detecting(temp, g_defect_data_cam2, 0, 0, 0, 0, 0);
    }
    catch (const std::exception& e)
    {
        //LogWriterInit("函数【MainFunction2】检测异常。");
        //LogWriterInit(e.what());
        /*try
        {
            ImgSave(temp);
        }
        catch (const std::exception&)
        {
            LogWriterInit("函数【MainFunction2】存图异常。");
            return StatusCode::ROI_IMG_EMPTY;
        }*/
        return StatusCode::SUCCESS;
    }

    // 将缺陷位置信息偏移到原图上
    defect_num = std::min(20, int(g_defect_data_cam2.size()));
    try
    {
        // DefectDataCSharp* defect_arr;
        // 构建C#返回值
        for (size_t j = 0; j < defect_num; ++j)
        {
            // g_defect_data_cam2[j].y += y;
            defect_data_ptr[j].defect_id = g_defect_data_cam2[j].defect_id;
            std::strcpy(defect_data_ptr[j].defect_name, g_defect_data_cam2[j].defect_name.c_str());
            defect_data_ptr[j].x = g_defect_data_cam2[j].x + x;
            defect_data_ptr[j].y = g_defect_data_cam2[j].y + y;
            defect_data_ptr[j].w = g_defect_data_cam2[j].w;
            defect_data_ptr[j].h = g_defect_data_cam2[j].h;
            defect_data_ptr[j].score = g_defect_data_cam2[j].score;
        }
    }
    catch (const std::exception& e)
    {
        //LogWriterInit(e.what());
        return StatusCode::CSHARPFAIL;
    }

    return ret;
}

DLL_API void CATL_JUANRAO::GlobalUninit()
{
    try
    {
        // 释放检测资源
        if (juanrao_init)
        {       
            delete juanrao;
            juanrao = nullptr;
            juanrao_init = false;
        }
        // 释放检测资源
        if (juanrao_init1)
        {
            delete juanrao1;
            juanrao1 = nullptr;
            juanrao_init1 = false;
        }
        /*if (juanrao_init2)
        {
            delete juanrao2;
            juanrao2 = nullptr;
            juanrao_init2 = false;
        }
        if (juanrao_init3)
        {
            delete juanrao3;
            juanrao3 = nullptr;
            juanrao_init3 = false;
        }*/
    }
    catch (const std::exception& e)
    {
        //LogWriterInit(e.what());
    }
}
