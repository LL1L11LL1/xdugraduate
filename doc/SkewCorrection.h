#pragma once

/**
 * @file SkewCorrection.h
 * @brief Skew correction module header file
 * 
 * This module provides document image skew correction functionality, rotating images based on detected skew angle,
 * restoring them to horizontal state. Also handles boundary issues after rotation to ensure image integrity.
 * 
 * 功能：本模块提供文档图像倾斜校正功能，根据检测到的倾斜角度旋转图像，
 *      将其恢复到水平状态。同时处理旋转后的边界问题，确保图像完整性。
 */

#include <opencv2/opencv.hpp>

/**
 * @class SkewCorrection
 * @brief Document image skew correction class
 * 
 * Provides static methods for rotating images based on detected skew angle.
 * Automatically calculates rotated image size to avoid content loss.
 * All methods are static, no instantiation required.
 * 
 * 功能：文档图像倾斜校正类，提供基于检测到的倾斜角度旋转图像的静态方法。
 * 自动计算旋转后的图像尺寸，避免内容丢失。
 * 所有方法均为静态方法，无需实例化即可直接调用。
 */
class SkewCorrection {
public:
    /**
     * @brief Perform skew correction
     * 
     * Rotate image based on detected skew angle to restore it to horizontal state.
     * 
     * Processing flow:
     * 1. Calculate rotation center (image center point)
     * 2. Get rotation matrix
     * 3. Calculate rotated image size (avoid cropping)
     * 4. Adjust translation component of rotation matrix
     * 5. Perform affine transform (rotation)
     * 
     * Rotation formulas:
     * New width = original height × |sin(θ)| + original width × |cos(θ)|
     * New height = original height × |cos(θ)| + original width × |sin(θ)|
     * 
     * @param image Input original image (color or grayscale)
     * @param angle Skew angle (degrees), positive value means clockwise rotation, negative means counterclockwise
     * @return cv::Mat Corrected image, size may be larger than original
     * 
     * 功能：执行倾斜校正，根据检测到的倾斜角度旋转图像，将其恢复到水平状态。
     * 
     * 处理流程：
     * 1. 计算旋转中心（图像中心点）
     * 2. 获取旋转矩阵
     * 3. 计算旋转后的图像尺寸（避免裁剪）
     * 4. 调整旋转矩阵的平移分量
     * 5. 执行仿射变换（旋转）
     * 
     * 旋转公式：
     * 新宽度 = 原始高度 × |sin(θ)| + 原始宽度 × |cos(θ)|
     * 新高度 = 原始高度 × |cos(θ)| + 原始宽度 × |sin(θ)|
     * 
     * @param image 输入的原始图像（彩色或灰度）
     * @param angle 倾斜角度（度），正值表示顺时针旋转，负值表示逆时针旋转
     * @return 校正后的图像，尺寸可能大于原始图像
     */
    static cv::Mat correctSkew(const cv::Mat& image, double angle);
};
