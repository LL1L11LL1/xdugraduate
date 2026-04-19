#pragma once

/**
 * @file ImageProcessor.h
 * @brief 图像预处理模块头文件
 * 
 * 本模块提供文档图像的预处理功能，包括灰度化、二值化、去噪和边缘提取。
 * 这些预处理步骤是倾斜检测和校正的基础。
 */

#include <opencv2/opencv.hpp>

/**
 * @class ImageProcessor
 * @brief 图像预处理类
 * 
 * 提供静态方法用于文档图像的预处理，所有方法均为静态方法，
 * 无需实例化即可直接调用。
 */
class ImageProcessor {
public:
    /**
     * @brief 完整的预处理流程
     * 
     * 依次执行：灰度化 -> 去噪 -> 二值化 -> 边缘提取
     * 
     * @param image 输入的原始图像
     * @return cv::Mat 预处理后的边缘图像
     */
    static cv::Mat preprocess(const cv::Mat& image);
    
    /**
     * @brief 图像灰度化
     * 
     * 将彩色图像转换为灰度图像。如果输入已经是灰度图像，则直接返回副本。
     * 
     * @param image 输入图像（彩色或灰度）
     * @return cv::Mat 灰度图像
     */
    static cv::Mat grayscale(const cv::Mat& image);
    
    /**
     * @brief 图像二值化
     * 
     * 使用自适应高斯阈值进行二值化，适合光照不均匀的文档图像。
     * 使用THRESH_BINARY_INV模式，使文字为白色，背景为黑色。
     * 
     * @param image 输入的灰度图像
     * @return cv::Mat 二值化图像
     */
    static cv::Mat binarize(const cv::Mat& image);
    
    /**
     * @brief 图像去噪
     * 
     * 使用高斯模糊去除图像噪声，同时保留边缘信息。
     * 核大小为5x5，标准差自动计算。
     * 
     * @param image 输入图像
     * @return cv::Mat 去噪后的图像
     */
    static cv::Mat denoise(const cv::Mat& image);
    
    /**
     * @brief 边缘提取
     * 
     * 使用Canny算法提取图像边缘，用于后续的直线检测。
     * 低阈值50，高阈值150。
     * 
     * @param image 输入的二值化图像
     * @return cv::Mat 边缘图像
     */
    static cv::Mat edgeExtract(const cv::Mat& image);
};
