#pragma once

/**
 * @file SkewDetection.h
 * @brief Skew detection module header file
 * 
 * This module provides multiple document image skew angle detection algorithms, including:
 * - Hough transform: Calculate skew angle by detecting lines, high precision
 * - Projection analysis: Determine skew angle by analyzing projection energy
 * - Least squares: Calculate skew angle by fitting edge points
 * - Radon transform: Calculate skew angle using Radon transform
 * 
 * These algorithms are suitable for different scenarios of document image skew detection.
 * 
 * 功能：本模块提供多种文档图像倾斜角度检测算法，包括：
 * - 霍夫变换：通过检测直线计算倾斜角度，精度高
 * - 投影分析：通过分析投影能量确定倾斜角度
 * - 最小二乘法：通过拟合边缘点计算倾斜角度
 * - 拉东变换：使用拉东变换计算倾斜角度
 * 
 * 这些算法适用于不同场景的文档图像倾斜检测。
 */

#include <opencv2/opencv.hpp>

/**
 * @class SkewDetection
 * @brief Document image skew detection class
 * 
 * Provides four skew detection algorithms, users can choose the appropriate method based on document type and precision requirements.
 * All methods are static, no instantiation required.
 * 
 * 功能：文档图像倾斜检测类，提供四种倾斜检测算法，用户可以根据文档类型和精度要求选择合适的方法。
 * 所有方法均为静态方法，无需实例化即可直接调用。
 */
class SkewDetection {
public:
    /**
     * @brief Skew detection algorithm enum
     * 
     * Defines four available skew detection algorithms:
     * - HOUGH_TRANSFORM: Hough transform, suitable for complex layouts
     * - PROJECTION_ANALYSIS: Projection analysis, suitable for simple text
     * - LEAST_SQUARES: Least squares, suitable for images with clear edges
     * - RADON_TRANSFORM: Radon transform, suitable for various document types
     * 
     * 功能：倾斜检测算法枚举，定义了四种可用的倾斜检测算法：
     * - HOUGH_TRANSFORM: 霍夫变换，适合复杂布局
     * - PROJECTION_ANALYSIS: 投影分析，适合简单文本
     * - LEAST_SQUARES: 最小二乘法，适合边缘清晰的图像
     * - RADON_TRANSFORM: 拉东变换，适合各种文档类型
     */
    enum Algorithm {
        HOUGH_TRANSFORM,      ///< 霍夫变换算法
        PROJECTION_ANALYSIS,  ///< 投影分析算法
        LEAST_SQUARES,        ///< 最小二乘法算法
        RADON_TRANSFORM       ///< 拉东变换算法
    };

    /**
     * @brief Unified skew detection interface
     * 
     * Call corresponding detection method based on specified algorithm type.
     * 
     * @param image Preprocessed edge image
     * @param algorithm Detection algorithm to use (HOUGH_TRANSFORM/PROJECTION_ANALYSIS/LEAST_SQUARES/RADON_TRANSFORM)
     * @return double Detected skew angle (degrees), positive value means clockwise skew, negative means counterclockwise skew
     * 
     * 功能：统一的倾斜检测接口，根据指定的算法类型调用相应的检测方法。
     * @param image 预处理后的边缘图像
     * @param algorithm 使用的检测算法 (HOUGH_TRANSFORM/PROJECTION_ANALYSIS/LEAST_SQUARES/RADON_TRANSFORM)
     * @return 检测到的倾斜角度（度），正值表示顺时针倾斜，负值表示逆时针倾斜
     */
    static double detectSkew(const cv::Mat& image, Algorithm algorithm);
    
    /**
     * @brief Hough transform skew detection
     * 
     * Use Hough transform to detect lines in image, calculate document skew angle by analyzing line angles.
     * 
     * Algorithm principle:
     * 1. Use HoughLines to detect all lines in edge image
     * 2. Extract angle (theta) for each line
     * 3. Convert angles to degrees and filter valid angles (-45° to 45°)
     * 4. Take median as final skew angle
     * 
     * Advantages: High precision, suitable for complex layout documents
     * Disadvantages: High computational cost
     * 
     * @param image Edge image (from ImageProcessor::edgeExtract)
     * @return double Skew angle (degrees)
     * 
     * 功能：霍夫变换倾斜检测，使用霍夫变换检测图像中的直线，通过分析直线角度计算文档倾斜角度。
     * 
     * 算法原理：
     * 1. 使用 HoughLines 检测边缘图像中的所有直线
     * 2. 提取每条直线的角度（theta）
     * 3. 将角度转换为度数并过滤有效角度（-45° 到 45°）
     * 4. 取中位数作为最终倾斜角度
     * 
     * 优点：精度高，适合复杂布局文档
     * 缺点：计算成本高
     * 
     * @param image 边缘图像（来自 ImageProcessor::edgeExtract）
     * @return 倾斜角度（度）
     */
    static double houghTransform(const cv::Mat& image);
    
    /**
     * @brief Projection analysis skew detection
     * 
     * Calculate projection energy of image at different angles, find angle with maximum projection energy as skew angle.
     * 
     * Algorithm principle:
     * 1. Rotate image in -45° to 45° range with 0.5° step
     * 2. For each angle, calculate vertical projection energy (sum of squares of pixel values)
     * 3. Angle with maximum projection energy is the skew angle
     * 
     * Advantages: Suitable for documents with obvious text lines
     * Disadvantages: High computational cost, angle precision limited by step size
     * 
     * @param image Edge image or binary image
     * @return double Skew angle (degrees)
     * 
     * 功能：投影分析倾斜检测，计算不同角度下图像的投影能量，找到投影能量最大的角度作为倾斜角度。
     * 
     * 算法原理：
     * 1. 在 -45° 到 45° 范围内以 0.5° 步长旋转图像
     * 2. 对每个角度，计算垂直投影能量（像素值的平方和）
     * 3. 投影能量最大的角度即为倾斜角度
     * 
     * 优点：适合有明显文本行的文档
     * 缺点：计算成本高，角度精度受步长限制
     * 
     * @param image 边缘图像或二值图像
     * @return 倾斜角度（度）
     */
    static double projectionAnalysis(const cv::Mat& image);
    
    /**
     * @brief Least squares skew detection
     * 
     * Fit edge points using least squares method, calculate slope of best-fit line, then derive skew angle.
     * 
     * Algorithm principle:
     * 1. Extract coordinates of all edge points
     * 2. Fit line using least squares: y = kx + b
     * 3. Calculate skew angle from slope k: angle = arctan(k) * 180/π
     * 
     * Advantages: Fast calculation speed
     * Disadvantages: Sensitive to noise, requires clear edges
     * 
     * @param image Edge image
     * @return double Skew angle (degrees)
     * 
     * 功能：最小二乘法倾斜检测，使用最小二乘法拟合边缘点，计算最佳拟合直线的斜率，然后推导出倾斜角度。
     * 
     * 算法原理：
     * 1. 提取所有边缘点的坐标
     * 2. 使用最小二乘法拟合直线：y = kx + b
     * 3. 从斜率 k 计算倾斜角度：angle = arctan(k) * 180/π
     * 
     * 优点：计算速度快
     * 缺点：对噪声敏感，需要清晰的边缘
     * 
     * @param image 边缘图像
     * @return 倾斜角度（度）
     */
    static double leastSquares(const cv::Mat& image);
    
    /**
     * @brief Radon transform skew detection
     * 
     * Use Radon transform to detect the dominant orientation in the image.
     * 
     * Algorithm principle:
     * 1. Apply Radon transform to the edge image
     * 2. Find the angle corresponding to the maximum value in the Radon transform
     * 3. Convert this angle to the skew angle
     * 
     * Advantages: Robust to noise, works well for various document types
     * Disadvantages: Higher computational cost
     * 
     * @param image Edge image
     * @return double Skew angle (degrees)
     * 
     * 功能：拉东变换倾斜检测，使用拉东变换检测图像中的主要方向。
     * 
     * 算法原理：
     * 1. 对边缘图像应用拉东变换
     * 2. 找到拉东变换中最大值对应的角度
     * 3. 将此角度转换为倾斜角度
     * 
     * 优点：对噪声鲁棒，适用于各种文档类型
     * 缺点：计算成本较高
     * 
     * @param image 边缘图像
     * @return 倾斜角度（度）
     */
    static double radonTransform(const cv::Mat& image);
};
