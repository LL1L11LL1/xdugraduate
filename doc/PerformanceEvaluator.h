#pragma once

/**
 * @file PerformanceEvaluator.h
 * @brief Performance evaluation module header file
 * 
 * This module provides performance evaluation functionality for document image skew detection algorithms,
 * including detection accuracy evaluation (angle error) and processing speed evaluation (processing time).
 * Supports batch testing and single image testing.
 * 
 * 功能：本模块提供文档图像倾斜检测算法的性能评估功能，
 *      包括检测精度评估（角度误差）和处理速度评估（处理时间）。
 *      支持批量测试和单图像测试。
 */

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>

/**
 * @class PerformanceEvaluator
 * @brief Performance evaluation class
 * 
 * Provides static methods for evaluating performance metrics of skew detection algorithms.
 * Can evaluate processing time and angle error for a single image,
 * or batch evaluate performance for an entire test set.
 * All methods are static, no instantiation required.
 * 
 * 功能：性能评估类，提供评估倾斜检测算法性能指标的静态方法。
 * 可以评估单图像的处理时间和角度误差，
 * 或批量评估整个测试集的性能。
 * 所有方法均为静态方法，无需实例化即可直接调用。
 */
class PerformanceEvaluator {
public:
    /**
     * @brief Evaluation result structure
     * 
     * Stores results of a single evaluation, including angle error and processing time.
     * 
     * 功能：评估结果结构体，存储单次评估的结果，包括角度误差和处理时间。
     */
    struct Result {
        double angleError;      ///< 角度误差（度），与真实角度的差值
        double processingTime;  ///< 处理时间（毫秒），包括预处理、检测和校正
    };

    /**
     * @brief Batch evaluate test set
     * 
     * Perform batch testing on all images in the specified directory, calculate average angle error and processing time.
     * Supported image formats: .jpg, .png
     * 
     * Evaluation metrics:
     * - Average angle error: average of angle errors for all images
     * - Average processing time: average of processing times for all images
     * 
     * @param testDir Test image directory path
     * @param algorithmName Algorithm name ("hough"/"projection"/"leastSquares")
     * 
     * 功能：批量评估测试集，对指定目录中的所有图像执行批量测试，计算平均角度误差和处理时间。
     * 支持的图像格式：.jpg, .png
     * 
     * 评估指标：
     * - 平均角度误差：所有图像角度误差的平均值
     * - 平均处理时间：所有图像处理时间的平均值
     * 
     * @param testDir 测试图像目录路径
     * @param algorithmName 算法名称 ("hough"/"projection"/"leastSquares")
     */
    static void evaluate(const std::string& testDir, const std::string& algorithmName);
    
    /**
     * @brief Single image evaluation
     * 
     * Perform skew detection on a single image and return evaluation results.
     * Automatically times the complete processing flow: preprocessing, detection, correction.
     * 
     * @param image Input original image
     * @param algorithm Algorithm type (SkewDetection::Algorithm enum value)
     * @return Result Evaluation result, containing angle error and processing time
     * 
     * 功能：单图像评估，对单个图像执行倾斜检测并返回评估结果。
     * 自动计时完整处理流程：预处理、检测、校正。
     * 
     * @param image 输入的原始图像
     * @param algorithm 算法类型（SkewDetection::Algorithm 枚举值）
     * @return 评估结果，包含角度误差和处理时间
     */
    static Result evaluateSingleImage(const cv::Mat& image, int algorithm);
};
