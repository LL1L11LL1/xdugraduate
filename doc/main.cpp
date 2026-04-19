#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include "ImageProcessor.h"
#include "SkewDetection.h"
#include "SkewCorrection.h"
#include "PerformanceEvaluator.h"

using namespace cv;
using namespace std;
using namespace chrono;

// 全局变量
Mat g_originalImage;    // 原始图像
Mat g_correctedImage;   // 校正后的图像
int g_algorithm = 0;    // 当前使用的算法 (0=霍夫变换, 1=投影分析, 2=最小二乘法, 3=拉东变换)

// 算法名称
const string algorithmNames[] = {"Hough Transform", "Projection Analysis", "Least Squares", "Radon Transform"};

/**
 * @brief Draw performance information on image
 *
 * Draw algorithm name, detected angle, processing time on image.
 * 
 * @param image Image to draw on
 * @param algorithm Algorithm index
 * @param angle Detected angle
 * @param time Processing time (milliseconds)
 * @return Mat Image with information drawn
 * 
 * 功能：在图像上绘制算法名称、检测到的角度和处理时间等性能信息
 */
Mat drawPerformanceInfo(Mat image, int algorithm, double angle, double time) {
    Mat result = image.clone();
    
    // Set font
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;
    
    // Draw algorithm name
    string algoName = "Algorithm: " + algorithmNames[algorithm];
    putText(result, algoName, Point(20, 40), fontFace, fontScale, Scalar(0, 0, 255), thickness);
    
    // Draw detected angle
    string angleStr = "Angle: " + to_string(angle).substr(0, 5) + " degrees";
    putText(result, angleStr, Point(20, 80), fontFace, fontScale, Scalar(0, 0, 255), thickness);
    
    // Draw processing time
    string timeStr = "Time: " + to_string(time).substr(0, 5) + "ms";
    putText(result, timeStr, Point(20, 120), fontFace, fontScale, Scalar(0, 0, 255), thickness);
    
    // Draw operation hints
    string hint = "S:Save O:Reload ESC:Exit";
    putText(result, hint, Point(20, result.rows - 20), fontFace, fontScale, Scalar(0, 255, 0), thickness);
    
    return result;
}

/**
 * @brief Evaluate all algorithms and display comparison results
 *
 * Evaluate all four skew detection algorithms on current image,
 * calculate angle error and processing time for each algorithm,
 * and create a table image to display comparison results.
 * 
 * 功能：评估所有四种倾斜检测算法，计算每种算法的角度误差和处理时间，
 *      并创建表格图像显示比较结果
 */
void evaluateAllAlgorithms() {
    // 检查原始图像是否存在
    if (g_originalImage.empty()) {
        cout << "Please load image first" << endl;
        return;
    }

    // 存储每种算法的评估结果
    struct AlgorithmResult {
        string name;        // 算法名称
        double angleError;  // 角度误差
        double processingTime;  // 处理时间
        double detectedAngle;   // 检测到的角度
    };

    vector<AlgorithmResult> results;  // 存储所有算法的结果

    // 评估每种算法
    for (int algo = 0; algo < 4; algo++) {
        // 记录开始时间
        auto start = high_resolution_clock::now();

        // 执行图像预处理
        Mat processed = ImageProcessor::preprocess(g_originalImage);

        // 执行倾斜检测
        double angle = SkewDetection::detectSkew(processed, static_cast<SkewDetection::Algorithm>(algo));

        // 记录结束时间
        auto end = high_resolution_clock::now();
        double processingTime = duration<double, milli>(end - start).count();

        // 计算角度误差（假设真实角度为0度）
        double angleError = abs(angle);

        // 存储结果
        AlgorithmResult result;
        result.name = algorithmNames[algo];
        result.angleError = angleError;
        result.processingTime = processingTime;
        result.detectedAngle = angle;
        results.push_back(result);

        // 输出结果
        cout << "=================================" << endl;
        cout << "Algorithm: " << result.name << endl;
        cout << "Detected angle: " << result.detectedAngle << " degrees" << endl;
        cout << "Angle error: " << result.angleError << " degrees" << endl;
        cout << "Processing time: " << result.processingTime << " milliseconds" << endl;
    }
    cout << "=================================" << endl;

    // 创建表格图像
    int width = 800;
    int height = 500;
    Mat tableImage(height, width, CV_8UC3, Scalar(255, 255, 255));

    // 设置字体
    int fontFace = FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.7;
    int thickness = 2;

    // 绘制标题
    string title = "Algorithm Performance Comparison";
    putText(tableImage, title, Point(width/2 - 120, 50), fontFace, 1.2, Scalar(0, 0, 0), 3);

    // 绘制表格表头
    putText(tableImage, "Algorithm", Point(100, 100), fontFace, fontScale, Scalar(0, 0, 0), thickness);
    putText(tableImage, "Angle (deg)", Point(250, 100), fontFace, fontScale, Scalar(0, 0, 0), thickness);
    putText(tableImage, "Error (deg)", Point(450, 100), fontFace, fontScale, Scalar(0, 0, 0), thickness);
    putText(tableImage, "Time (ms)", Point(600, 100), fontFace, fontScale, Scalar(0, 0, 0), thickness);

    // 绘制表格线
    line(tableImage, Point(50, 120), Point(width-50, 120), Scalar(0, 0, 0), 2);

    // 绘制数据行
    for (int i = 0; i < results.size(); i++) {
        int y = 160 + i * 80;
        
        // 绘制算法名称
        putText(tableImage, results[i].name, Point(100, y), fontFace, fontScale, Scalar(0, 0, 255), thickness);
        
        // 绘制检测到的角度
        string angleStr = to_string(results[i].detectedAngle).substr(0, 6);
        putText(tableImage, angleStr, Point(280, y), fontFace, fontScale, Scalar(0, 0, 0), thickness);
        
        // 绘制角度误差
        string errorStr = to_string(results[i].angleError).substr(0, 6);
        putText(tableImage, errorStr, Point(480, y), fontFace, fontScale, Scalar(0, 0, 0), thickness);
        
        // 绘制处理时间
        string timeStr = to_string(results[i].processingTime).substr(0, 6);
        putText(tableImage, timeStr, Point(630, y), fontFace, fontScale, Scalar(0, 0, 0), thickness);
        
        // 绘制分隔线
        if (i < results.size() - 1) {
            line(tableImage, Point(50, y + 40), Point(width-50, y + 40), Scalar(0, 0, 0), 1);
        }
    }

    // 绘制结论
    // 找出角度误差最小的算法
    int bestAccuracyIndex = 0;
    double minError = results[0].angleError;
    for (int i = 1; i < results.size(); i++) {
        if (results[i].angleError < minError) {
            minError = results[i].angleError;
            bestAccuracyIndex = i;
        }
    }

    // 找出处理时间最短的算法
    int bestSpeedIndex = 0;
    double minTime = results[0].processingTime;
    for (int i = 1; i < results.size(); i++) {
        if (results[i].processingTime < minTime) {
            minTime = results[i].processingTime;
            bestSpeedIndex = i;
        }
    }

    // 绘制结论
    string bestAccuracy = "Best Accuracy: " + results[bestAccuracyIndex].name;
    string bestSpeed = "Fastest: " + results[bestSpeedIndex].name;
    putText(tableImage, bestAccuracy, Point(100, height - 60), fontFace, fontScale, Scalar(0, 255, 0), thickness);
    putText(tableImage, bestSpeed, Point(400, height - 60), fontFace, fontScale, Scalar(0, 255, 0), thickness);

    // 显示表格图像
    namedWindow("Algorithm Comparison", WINDOW_NORMAL);
    imshow("Algorithm Comparison", tableImage);

    // 保存表格图像
    string filename = "algorithm_comparison.jpg";
    imwrite(filename, tableImage);
    cout << "Algorithm comparison table saved as: " << filename << endl;
}

/**
 * @brief Algorithm selection callback function
 *
 * Called when user changes algorithm selection via trackbar.
 * Re-performs preprocessing and skew detection, updates corrected image display.
 *
 * @param pos Trackbar position (0, 1, 2, or 3)
 * @param userdata User data (not used)
 * 
 * 功能：算法选择回调函数，当用户通过滑块更改算法选择时调用，
 *      重新执行预处理和倾斜检测，更新校正后的图像显示
 */
void onAlgorithmChange(int pos, void* userdata) {
    // 检查原始图像是否存在
    if (g_originalImage.empty()) return;

    // 更新全局算法变量
    g_algorithm = pos;

    // 记录开始时间
    auto start = high_resolution_clock::now();

    // 执行图像预处理
    Mat processed = ImageProcessor::preprocess(g_originalImage);

    // 执行倾斜检测
    double angle = SkewDetection::detectSkew(processed, static_cast<SkewDetection::Algorithm>(g_algorithm));

    // 执行倾斜校正
    g_correctedImage = SkewCorrection::correctSkew(g_originalImage, angle);

    // 记录结束时间
    auto end = high_resolution_clock::now();
    double processingTime = duration<double, milli>(end - start).count();

    // 在校正后的图像上绘制性能信息
    Mat displayImage = drawPerformanceInfo(g_correctedImage, g_algorithm, angle, processingTime);

    // 更新显示
    imshow("Corrected", displayImage);

    // 输出当前算法和检测到的角度
    cout << "Current algorithm: " << algorithmNames[g_algorithm] << endl;
    cout << "Detected angle: " << angle << " degrees" << endl;
    cout << "Processing time: " << processingTime << " milliseconds" << endl;
}

/**
 * @brief Open and process image
 *
 * Load test.jpg image, perform complete processing flow:
 * 1. Read image
 * 2. Preprocess (grayscale, denoise, binarize, edge extract)
 * 3. Skew detection
 * 4. Skew correction
 * 5. Create display windows and trackbar
 * 6. Display original and corrected images
 * 7. Automatically evaluate all algorithms and display comparison
 * 
 * 功能：打开并处理图像，执行完整的处理流程：
 *      1. 读取图像
 *      2. 预处理（灰度化、去噪、二值化、边缘提取）
 *      3. 倾斜检测
 *      4. 倾斜校正
 *      5. 创建显示窗口和滑块
 *      6. 显示原始和校正后的图像
 *      7. 自动评估所有算法并显示比较结果
 */
void openImage() {
    // 图像文件名
    string filename = "test.jpg";

    // 读取图像
    g_originalImage = imread(filename);

    // 检查图像是否成功读取
    if (g_originalImage.empty()) {
        cout << "Cannot open or find image: " << filename << endl;
        cout << "Please ensure test.jpg exists in program directory" << endl;
        return;
    }

    // 记录开始时间
    auto start = high_resolution_clock::now();

    // 执行图像预处理
    Mat processed = ImageProcessor::preprocess(g_originalImage);

    // 执行倾斜检测（使用当前算法）
    double angle = SkewDetection::detectSkew(processed, static_cast<SkewDetection::Algorithm>(g_algorithm));

    // 执行倾斜校正
    g_correctedImage = SkewCorrection::correctSkew(g_originalImage, angle);

    // 记录结束时间
    auto end = high_resolution_clock::now();
    double processingTime = duration<double, milli>(end - start).count();

    // 在校正后的图像上绘制性能信息
    Mat displayImage = drawPerformanceInfo(g_correctedImage, g_algorithm, angle, processingTime);

    // 创建显示窗口
    namedWindow("Original", WINDOW_NORMAL);   // 原始图像窗口
    namedWindow("Corrected", WINDOW_NORMAL);  // 校正图像窗口

    // 显示图像
    imshow("Original", g_originalImage);
    imshow("Corrected", displayImage);

    // 创建算法选择滑块
    // 滑块范围：0-3，对应四种算法
    createTrackbar("Algorithm: 0=Hough, 1=Projection, 2=LeastSquares, 3=Radon", "Corrected", &g_algorithm, 3, onAlgorithmChange);

    // 输出图像加载成功信息
    cout << "Image loaded successfully: " << filename << endl;
    cout << "Image size: " << g_originalImage.cols << "x" << g_originalImage.rows << endl;
    cout << "Current algorithm: " << algorithmNames[g_algorithm] << endl;
    cout << "Detected angle: " << angle << " degrees" << endl;
    cout << "Processing time: " << processingTime << " milliseconds" << endl;

    // 自动评估所有算法并显示比较结果
    cout << "\nAutomatically evaluating all algorithms..." << endl;
    evaluateAllAlgorithms();
}

/**
 * @brief Save corrected image
 *
 * Save corrected image as corrected_output.jpg file.
 * If no corrected image exists, output error message.
 * 
 * 功能：保存校正后的图像为 corrected_output.jpg 文件，
 *      如果没有校正后的图像，则输出错误信息
 */
void saveImage() {
    // 检查校正后的图像是否存在
    if (g_correctedImage.empty()) {
        cout << "No corrected image to save" << endl;
        return;
    }

    // 保存图像
    string filename = "corrected_output.jpg";
    imwrite(filename, g_correctedImage);
    cout << "Corrected image saved as: " << filename << endl;
}

/**
 * @brief Main function
 *
 * Program entry point, execution flow:
 * 1. Open and process image
 * 2. Enter main loop, wait for user input
 * 3. Perform corresponding operations based on user keystrokes
 * 4. Release resources and exit
 *
 * Supported keyboard operations:
 * - ESC: Exit program
 * - S/s: Save corrected image
 * - O/o: Reload image
 *
 * @return int Program exit code (0 for success)
 * 
 * 功能：程序入口点，执行流程：
 *      1. 打开并处理图像
 *      2. 进入主循环，等待用户输入
 *      3. 根据用户按键执行相应操作
 *      4. 释放资源并退出
 * 
 * 支持的键盘操作：
 * - ESC: 退出程序
 * - S/s: 保存校正后的图像
 * - O/o: 重新加载图像
 */
int main() {
    // 打开并处理图像
    openImage();

    // 如果图像加载失败，直接退出
    if (g_originalImage.empty()) {
        cout << "Program exit: Cannot load image" << endl;
        return -1;
    }

    // 主循环：等待用户输入
    cout << endl << "Operation instructions:" << endl;
    cout << "  ESC - Exit program" << endl;
    cout << "  S   - Save corrected image" << endl;
    cout << "  O   - Reload image" << endl;
    cout << "  Use trackbar to switch between different skew detection algorithms" << endl << endl;

    while (true) {
        // 等待键盘输入
        int key = waitKey(0);

        // 根据按键执行相应操作
        if (key == 27) {
            // ESC键：退出程序
            break;
        } else if (key == 's' || key == 'S') {
            // S键：保存图像
            saveImage();
        } else if (key == 'o' || key == 'O') {
            // O键：重新加载图像
            openImage();
        }
    }

    // 销毁所有窗口
    destroyAllWindows();

    cout << "Program exited" << endl;
    return 0;
}
