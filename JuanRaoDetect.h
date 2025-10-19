#ifndef JUANRAO_DETECT_H
#define JUANRAO_DETECT_H


// 添加要在此处预编译的标头
#include "framework.h"
#include "CommonData.h"
#include  "opencv2/core/core.hpp"


namespace CATL_JUANRAO
{
#ifndef DLL_API
    #ifdef CATL_JUANRAO_EXPORTS
        #define DLL_API extern "C" __declspec(dllexport)
    #else
        #define DLL_API extern "C" __declspec(dllimport)
    #endif
#endif 

    /************************************************************************************************************************
        * Description: AI程序的API初始化接口函数
        *
        * Input:       @param cameraType：    相机数量，当前最多支持2个相机同时检测， 说明：1-初始化一个相机模型，2-初始化两个
                                                  相机模型，初始化接口只调用一次
        *
        * Outpput:     无
        *
        * Return:      运行状态，0-成功，其他失败
        *
        * Author:      马某
        *
        * Data:        2023.01.08
        *************************************************************************************************************************/
    DLL_API StatusCode InitModel(int cameraType);

    /************************************************************************************************************************
        * Description: AI参数重新加载。
        *
        * Author:      BQ
        *
        * Data:        2023.12.20
        *************************************************************************************************************************/
    DLL_API void ReloadParams();

    /************************************************************************************************************************
        * Description: C++获取AI版本信息
        *
        * Return:      std::string类型版本信息
        *
        * Author:      BQ
        *
        * Data:        2024.3.11
        *************************************************************************************************************************/
    std::string GetVervionInfo();


    /************************************************************************************************************************
        * Description: C#获取AI版本信息
        *
        * Return:      char*类型版本信息
        *
        * Author:      BQ
        *
        * Data:        2024.3.11
        *************************************************************************************************************************/
    DLL_API char* GetVersionInfo();


    /************************************************************************************************************************
        * Description: 获取AI参数信息
        *
        * Return:
        *
        * Author:      BQ
        *
        * Data:        2024.4.9
        *************************************************************************************************************************/
    DLL_API AI_Params GetParamInfo();



    /*****************************   检测接口，针对贴胶机台，增加贴胶位置参数   *****************************/
#pragma region
    /************************************************************************************************************************
        * Description: 调用AI模型接口函数
        *
        * Input:       @param roi_img：输入待检测图像
        *              @param x:   待检测图像极片左上角x
        *              @param y:   待检测图像极片左上角y
        *              @param w:   待检测图像极片区域宽度w
        *              @param h:   待检测图像极片区域高度h
        *              @param y1:  贴胶区域上边缘y坐标
        *              @param y2:  贴胶区域下边缘y坐标
        *              @tape_flag: 是否存在贴胶标志位，0-没有贴胶，1-有贴胶，有贴胶则AI内部进行定位，过滤
        *
        * Outpput:     @param defect_data：AI检测结果向量，记录全部缺陷信息
        *
        * Return:      运行状态，0-成功，其他-异常
        *
        * Author:      马某
        *
        * Data:        2023.01.08
        *************************************************************************************************************************/
    //DLL_API StatusCode CatlDetect(const cv::Mat& roi_img, int x, int y, int w, int h, int y1, int y2, std::vector<DefectData>& defect_data, int tape_flag);
    DLL_API StatusCode CatlDetect(const cv::Mat& roi_img, int x, int y, int w, int h, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, std::vector<DefectData>& defect_data, int tape_flag);
    /************************************************************************************************************************
       * Description: 2号相机调用AI模型接口函数
       *
       * Input:       @param roi_img：输入待检测图像
       *              @param x:   待检测图像极片左上角x
       *              @param y:   待检测图像极片左上角y
       *              @param w:   待检测图像极片区域宽度w
       *              @param h:   待检测图像极片区域高度h
       *              @param XYdataUp:上膜区角点坐标
       *              @param XYdataDown:下膜区角点坐标
       *              @tape_flag: 标志位，0—常规类别检测；1—切断胶胶纸检测，2—拐角胶检测
       *
       * Outpput:     @param defect_data：AI检测结果向量，记录全部缺陷信息
       *
       * Return:      运行状态，0-成功，其他-异常
       *
       * Author:      马某
       *
       * Data:        2023.01.08
       *************************************************************************************************************************/
    DLL_API StatusCode CatlDetectCam2(const cv::Mat& roi_img, int x, int y, int w, int h, std::vector<std::pair<int, int>>& XYdataUp, std::vector<std::pair<int, int>>& XYdataDown, std::vector<DefectData>& defect_data, int tape_flag);

    /************************************************************************************************************************
      * Description: 调用AI模型接口函数 C#版本--【博拓机】
      *
      * Input:       @param src_img：待检测源图像
      *              @param x:   待检测图像极片左上角x
      *              @param y:   待检测图像极片左上角y
      *              @param w:   待检测图像极片区域宽度w
      *              @param h:   待检测图像极片区域高度h
      *              @tape_pos:  待检测图像中贴胶的坐标位置信息(每个贴胶的顶部y值和底部y值)，外部传递Length为4的贴胶数组，\
      *                          若top_y, bottom_y的值为-1，表示该索引位置胶数据信息无意义，AI内部过滤贴胶位置信息
      *
      * Outpput:     @param defect_data*：AI检测缺陷结果数组,外部构造长度为20的缺陷信息
      *              @param defect_num: 缺陷数量
      *
      * Return:      运行状态，0-成功，其他-异常
      *
      * Author:      马某
      *
      * Data:        2023.01.08
      *************************************************************************************************************************/
    DLL_API  StatusCode CatlDetectCSharp(unsigned char* r_img_ptr, unsigned char* g_img_ptr, unsigned char* b_img_ptr, int src_img_width, \
        int src_img_height, int x, int y, int w, int h, DefectDataCSharp* defect_data_ptr, int& defect_num, TapePosition* tape_pos);

    /************************************************************************************************************************
      * Description: 2号相机调用AI模型接口函数C#版本--【博拓机】
      *
      * Input:       @param src_img：待检测源图像
      *              @param x:   待检测图像极片左上角x
      *              @param y:   待检测图像极片左上角y
      *              @param w:   待检测图像极片区域宽度w
      *              @param h:   待检测图像极片区域高度h
      *              @param tape_pos:   待检测图像中贴胶的坐标位置信息(每个贴胶的顶部y值和底部y值)，外部传递Length为4的贴胶数组，\
      *                          若top_y, bottom_y的值为-1，表示该索引位置胶数据信息无意义，AI内部过滤贴胶位置信息
      *
      * Outpput:     @param defect_data：AI检测缺陷结果数组,外部构造长度为20的缺陷信息
      *              @param defect_num: 缺陷数量
      *
      * Return:      运行状态，0-成功，其他-异常
      *
      * Author:      马某
      *
      * Data:        2023.01.08
      *************************************************************************************************************************/
    DLL_API StatusCode CatlDetectCam2CSharp(unsigned char* r_img_ptr, unsigned char* g_img_ptr, unsigned char* b_img_ptr, int src_img_width, \
        int src_img_height, int x, int y, int w, int h, DefectDataCSharp* defect_data_ptr, int& defect_num, TapePosition* tape_pos);

#pragma endregion
    /************************************************************************************************************************
       * Description: 释放模型句柄，谨慎使用，释放的全局模型实例，释放后所有模型资源均被释放,最好在程序退出检测逻辑进行释放
       *
       * Input:       无
       *
       * Outpput:     无
       *
       * Return:      运行状态，0-成功，其他-异常
       *
       * Author:      马某
       *
       * Data:        2023.01.08
       *************************************************************************************************************************/
    DLL_API void GlobalUninit();
}





































#endif // !JUANRAO_DETECT_H
