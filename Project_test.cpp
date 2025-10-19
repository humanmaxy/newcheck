// Project_test.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//
#define DLL_API
#include "function.h"
#include <JuanraoDetect.h>
#include <thread>
#include <fstream>
using namespace CATL_JUANRAO;

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

void threadtest0(int i)
{
    std::cout << i << " Test Thread " << "\n";
}

void threadtest1(int i, int j)
{
    std::cout << i << " Test Thread2 "  << j << "\n";
}

void threadth1(cv::Mat roi_img, cv::Rect box_L, cv::Rect box_R, int xxx, int pos_x, int pos_xR, std::vector<DefectData>& cam1_result)
{
    double t1;  // 检测时间记录
    double t_cost;  // 检测时间统计
    /*while (1)
    {*/
        t1 = static_cast<double>(cv::getTickCount());
        //CatlDetect(roi_img, box_L.x + pos_x - xxx, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + xxx, roi_img.rows, 0, 0, cam1_result, 0);
        t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        std::cout << "AI1 processing cost(ms): " << t_cost << std::endl;
    /*}*/
}

void threadth2(cv::Mat roi_img, cv::Rect box_L, cv::Rect box_R, int xxx, int pos_x, int pos_xR, std::vector<DefectData> cam1_result)
{
    double t1;  // 检测时间记录
    double t_cost;  // 检测时间统计
    /*while (1)
    {*/
        t1 = static_cast<double>(cv::getTickCount());
        //CatlDetectCam2(roi_img, box_L.x + pos_x - xxx, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + xxx, roi_img.rows, 0, 0, cam1_result, 0);
        t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
        std::cout << "AI2 processing cost(ms): " << t_cost << std::endl;
   /* }*/
}

int main()
{
    double t1;  // 检测时间记录
    double t_cost;  // 检测时间统计
    std::vector<std::string> files; // 测试图像
    cv::Mat roi_img, roi_img1;                // ROI图像

    // 初始化AI模型,返回值为0表示成功，其他-失败，可对照状态码定位问题
    /*
        * NVIDIA_ERROR = -1,                // 无显卡或者显卡驱动异常
        * SUCCESS = 0,                      // 运行成功
        * ROI_IMG_EMPTY = 1,                // ROI图像为空
        * PREPROCESS_FAIL = 2,              // 图像预处理失败
        * INIT_DETECT_MODEL_FAILED = 3,     // 图像加载检测模型失败
        * INIT_CLASSIFY_MODEL_FAILED = 4,   // 图像加载分类模型失败
        * INIT_DETECT_HANDLE_FAILED = 5,    // 初始化检测实例失败
        * CLASSIFY_PREDICT_FAILED = 6,       // 分类失败
        * CLASS_INSTANCE_ALREADY_EXIST = 7,       // 检测已创建实例，无法再次初始化
        * CAMERA_TYPE_ERR = 8,                // 初始化相机参数错误，camera_type为1，但是调用了CatlDetectCam2接口
        * CSHARPFAIL = 9,                      // C#返回结果异常
        * TAPE_POSITION_ERR = 10                // 传递进来的贴胶位置存在问题
    */
    StatusCode state = InitModel(2);
    if (state)
    {
        std::cout << "初始化模型失败，错误码：" << state;
        return state;
    }
    std::cout << "Init model succeed." << std::endl;
    //SetVersionInfo();
    char* version = GetVersionInfo();
    for (size_t i = 0; i < strlen(version); i++)
    {
        std::cout << version[i];
    }
    //ReloadParams();
    // 获取测试图像
    std::string path_test = "";
    std::cout << "请输入图片路径：";
    std::cin >> path_test;
    std::string save_img_str = "";
    /*std::cout << "是否保存裁切图片（yes||no）：";
    std::cin >> save_img_str;*/
    getFiles(path_test, files);
    cv::Mat L_img, R_img, save_img;
    cv::Mat save_img2;
    double mx, mn, mxr, mnr;
    cv::Point pmx, pmn, pmx_r, pmn_r;
    cv::Rect box_L;//矩形对象
    cv::Rect box_R;//矩形对象
    bool L1 = false;
    bool caitu = false;
    bool zhuabian = false;
    cv::Mat temp; // 对检测结果进行保存
    cv::Mat test1;
    std::vector<DefectData> cam1_result;              // 相机1检测结果
    std::vector<DefectData> cam1_result2;
    std::string filename;

    //20241106 lilu 读取log坐标
    std::vector<std::vector<std::pair<int, int>>> XYdataUps;  std::vector<std::vector<std::pair<int, int>>> XYdataDowns;
    std::string log_path = path_test + "/AI_Log.txt";
    
    const char* log_name = log_path.c_str();
    std::ifstream log(log_name);
    std::string line;
    int index = 0;
    std::vector<std::pair<int, int>> XYdataUp;
    std::vector<std::pair<int, int>> XYdataDown;
    if (log)
    {
        while (getline(log, line))
        {
            std::string::size_type pos = line.find("x y");
            if (pos != std::string::npos)
            {
                std::string position = line.substr(pos + 1);
                std::string::size_type posD = position.find(",");
                std::string x = position.substr(posD-4, 4);
                std::string y = position.substr(posD + 1);
                std::pair<int, int> pos_point;
                pos_point.first = std::stoi(x);
                pos_point.second = std::stoi(y);
                if (index <= 3)
                {
                    XYdataUp.push_back(pos_point);
                }
                else
                {
                    XYdataDown.push_back(pos_point);
                }
                index++;
                if (index == 8)
                    index = 0;
                if (XYdataUp.size() == 4)
                {
                    XYdataUps.push_back(XYdataUp);
                    XYdataUp.clear();
                }
                if (XYdataDown.size() == 4)
                {
                    XYdataDowns.push_back(XYdataDown);
                    XYdataDown.clear();
                }
            }
        }



    }
    /*if ((XYdataDowns.size() != XYdataUps.size()) || (XYdataDowns.size() != files.size()))
    {
        std::cout <<"图片个数与log不对应" <<std::endl;
        return -1;
    }*/
    for (int i = 0; i < files.size(); ++i)
    {
        filename = getFilename(files[i]);
        roi_img = cv::imread(files[i], 1);
        //cv::rotate(roi_img, roi_img, 1);
        cv::Mat temp;
        if (caitu)
        {
            cv::namedWindow("img", CV_WINDOW_KEEPRATIO);//窗口
            cv::setMouseCallback("img", onmouse, &roi_img);
            cv::imshow("img", roi_img);
        }
        while (caitu)
        {
            if (drawing_box) {//不断更新正在画的矩形
                roi_img.copyTo(temp);//这句放在这里是保证了每次更新矩形框都是在没有原图的基础上更新矩形框。
                if (L1)
                {
                    rectangle(temp, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(0, 0, 255),3);
                    std::stringstream sx;
                    std::stringstream sy;
                    sx << mouseOn.x;
                    sy << mouseOn.y;
                    std::string txt = "right(" + sx.str() + "," + sy.str() + ")";
                    putText(temp, txt, mouseOn - cv::Point(2, 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                    cv::imshow("img", temp);//显示
                    cv::waitKey(10);
                }
           
                else
                {
                    rectangle(temp, cv::Point(box.x, box.y), cv::Point(box.x + box.width, box.y + box.height), cv::Scalar(255, 255, 255),3);
                    std::stringstream sx;
                    std::stringstream sy;
                    sx << mouseOn.x;
                    sy << mouseOn.y;
                    std::string txt = "lift(" + sx.str() + "," + sy.str() + ")";

                    putText(temp, txt, mouseOn - cv::Point(2, 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
                    cv::imshow("img", temp);//显示
                    cv::waitKey(10);
                }
            }
            if (cv::waitKey(10) == 49 && !L1) 
            {//检测是否有按下1键
                L1 = true;
                box_L = box;
                cv::imshow("img", roi_img);
                cv::waitKey(10);
                continue;
                }
            if (cv::waitKey(10) == 50) 
            {//检测是否有按下2键
                box_R = box;
                caitu = false;
                cv::destroyWindow("img");
                break;//退出循环
            }
        }
        //if (!zhuabian)
        //{
            //roi_img(box_L).copyTo(L_img);
            //if (L_img.channels() == 3)
            //    cv::cvtColor(L_img, L_img, cv::COLOR_BGR2GRAY);

            //cv::Mat reduceMat;
            //cv::reduce(L_img, reduceMat, 0, cv::REDUCE_AVG, CV_64F);
            //cv::Sobel(reduceMat, reduceMat, CV_64F, 1, 0);
            //int thresh = 20;
            //auto suppressedProfptr = reduceMat.ptr<double>();
            //int pos_x = 0;
            //double amplitude;
            //for (int i = 0; i < L_img.cols; ++i)
            //{
            //    if (abs(suppressedProfptr[i]) > thresh)
            //    {
            //        pos_x = i;
            //        break;
            //    }
            //}
            //cv::line(L_img, cv::Point(pos_x, 0), cv::Point(pos_x, L_img.rows - 1), -1, 11);
            //minMaxLoc(reduceMat, &mn, &mx, &pmn, &pmx);

            //roi_img(box_R).copyTo(R_img);
            //std::vector<cv::Mat> BGR;
            //cv::split(R_img, BGR);

            ///*if (R_img.channels() == 3)
            //    cv::cvtColor(R_img, R_img, cv::COLOR_BGR2GRAY);*/

            //cv::Mat reduceMatR;
            ////cv::reduce(R_img, reduceMatR, 0, cv::REDUCE_AVG, CV_64F);

            //cv::reduce(BGR[0], reduceMatR, 0, cv::REDUCE_AVG, CV_64F);

            //cv::Sobel(reduceMatR, reduceMatR, CV_64F, 1, 0);
            //suppressedProfptr = reduceMatR.ptr<double>();
            //int pos_xR = 0;
            //thresh = 5;
            //for (int i = reduceMatR.cols-1; i > 0; --i)
            //{
            //    if (abs(suppressedProfptr[i]) > thresh)
            //    {
            //    pos_xR = i;
            //    break;
            //    }
            //}
            //cv::line(R_img, cv::Point(pos_xR, 0), cv::Point(pos_xR, R_img.rows - 1), -1, 11);
            //minMaxLoc(reduceMatR, &mnr, &mxr, &pmn_r, &pmx_r);
            //zhuabian = true;
            //
            //std::cout << files[i] << std::endl;

            AI_Params aiparameter = GetParamInfo();

            t1 = static_cast<double>(cv::getTickCount());
            //::cvtColor(roi_img, roi_img, cv::COLOR_BGR2RGB);

            //左右抓到边之后分别外扩像素个数
            int pad = 100;
            //std::thread thread1(threadth1, roi_img, box_L, box_R, pad, pos_x, pos_xR, std::ref(cam1_result));
            // std::thread thread1(threadtest1, 0, 1);
            //std::thread thread2(threadth2, roi_img, box_L, box_R, pad, pos_x, pos_xR, cam1_result2);

            //thread1.join();
            //thread2.join();


            //if (x == 1)
            //{
            //    state = CatlDetect(roi_img, box_L.x + pos_x - 100, 0, ((box_R.x + pos_xR) - (box_L.x + pos_x)) / 2 + 80, roi_img.rows, cam1_result, 0);
            //    state = CatlDetect(roi_img, box_L.x + pos_x - 100 + ((box_R.x + pos_xR) - (box_L.x + pos_x)) / 2 + 80, 0,
            //        ((box_R.x + pos_xR) - (box_L.x + pos_x)) / 2, roi_img.rows, cam1_result2, 0);
            //}
            //else
            //{
            //    int xxx = 100;
            int y1, y2;
            std::cout << filename << " input y1 and y2" << std::endl;
            std::vector<std::pair<int, int>> XYdataUp;  std::vector<std::pair<int, int>> XYdataDown;
            std::string imgName = filename.substr(0, filename.size() - 4);
            int64 int_img = std::stoi(imgName);
            XYdataUp = XYdataUps[int_img - 1];
            XYdataDown = XYdataDowns[int_img - 1];
                       
            //std::cin >> y1 >> y2;
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3841, 4091, cam1_result, 0);
            //149
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3878, 4137, cam1_result, 0);
            //501
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3998, 4298, cam1_result, 0);
            // 1101test
            state = CatlDetect(roi_img, 1355, 3708, 5434, 655, XYdataUp, XYdataDown, cam1_result, 0);
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3877, 4131, cam1_result, 0);


            //std::cin >> y1 >> y2;
            //149
           // state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3878, 4137, cam1_result, 0);
            //501
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3998, 4298, cam1_result, 0);

            //state = CatlDetectCam2(roi_img, 1355, 3708, 5434, 655, 3870, 4140, cam1_result, 0);
            //state = CatlDetectCam2(roi_img, 1466, 0, 5843, 7628, 3841, 4095, cam1_result, 0);


            
            //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, 3750, 4100, cam1_result, 0);
            
            
             //state = CatlDetect(roi_img, box_L.x + pos_x - pad, 0, (box_R.x + pos_xR) - (box_L.x + pos_x) + pad, roi_img.rows, cam1_result, 0);
            
            


            t_cost = (static_cast<double>(cv::getTickCount()) - t1) / cv::getTickFrequency() * 1000;
            std::cout << "AI processing cost(ms): " << t_cost << std::endl;
            if (state)
            {
                std::cout << "检测失败，错误码：" << state << ",filename" << filename;
                continue;
            }
            temp = roi_img.clone();
            test1 = clahe_deal(temp);
            for (size_t i = 0; i < cam1_result.size(); ++i)
            {
                std::cout << "缺陷" << i << ":" << cam1_result[i].defect_name<<std::endl;
                cv::rectangle(temp, cv::Rect(cam1_result[i].x, cam1_result[i].y, cam1_result[i].w, cam1_result[i].h), cv::Scalar(25, 255, 255), 4);
                cv::putText(temp, cam1_result[i].defect_name + std::to_string(cam1_result[i].score), cv::Point(cam1_result[i].x, cam1_result[i].y + 50), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 25, 255), 2);
                
            }



            
            std::string full_path = files[i];
            std::string filename = getFilename(full_path);
            auto filen = filename.find_last_of(".");
            filename.replace(filen, 4, ".jpg");
            std::string sp = path_test + "_res";
            const char* res_path = (char*)(sp).data();
            if (_access(res_path, 0) == -1)
            {
                auto b = _mkdir(res_path);
            }
            std::string s;
            if (cam1_result.size() > 0)
            {
                s = path_test + "_res/NG/";
                
            }
            else
            {
                s = path_test + "_res/OK/";
            }
            res_path = (char*)(s).data();

            if (_access(res_path, 0) == -1)
            {
                auto b = _mkdir(res_path);
            }
            cv::imwrite(res_path + std::string("//") + filename, temp);

    }


    
    GlobalUninit();

}





// 运行程序: Ctrl + F5 或调试 >“开始执行(不调试)”菜单
// 调试程序: F5 或调试 >“开始调试”菜单

// 入门使用技巧: 
//   1. 使用解决方案资源管理器窗口添加/管理文件
//   2. 使用团队资源管理器窗口连接到源代码管理
//   3. 使用输出窗口查看生成输出和其他消息
//   4. 使用错误列表窗口查看错误
//   5. 转到“项目”>“添加新项”以创建新的代码文件，或转到“项目”>“添加现有项”以将现有代码文件添加到项目
//   6. 将来，若要再次打开此项目，请转到“文件”>“打开”>“项目”并选择 .sln 文件
