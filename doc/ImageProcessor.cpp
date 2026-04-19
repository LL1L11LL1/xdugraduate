#include "ImageProcessor.h"

/**
 * @brief Perform complete image preprocessing flow
 * 
 * Preprocessing steps:
 * 1. Grayscale: Convert color image to grayscale, reduce computation
 * 2. Denoise: Use Gaussian blur to remove scanning noise
 * 3. Binarization: Adaptive thresholding to enhance text contrast
 * 4. Edge extraction: Canny algorithm to detect edges for line detection
 * 
 * @param image Input original document image
 * @return cv::Mat Preprocessed edge image
 * 
 * 功能：执行完整的图像预处理流程，包括灰度化、去噪、二值化和边缘提取。
 * 预处理步骤：
 * 1. 灰度化：将彩色图像转换为灰度，减少计算量
 * 2. 去噪：使用高斯模糊去除扫描噪声
 * 3. 二值化：自适应阈值增强文本对比度
 * 4. 边缘提取：Canny算法检测边缘，为直线检测做准备
 * 
 * @param image 输入的原始文档图像
 * @return 预处理后的边缘图像
 */
cv::Mat ImageProcessor::preprocess(const cv::Mat& image) {
    // Step 1: Convert to grayscale
    cv::Mat gray = grayscale(image);
    
    // Step 2: Denoise
    cv::Mat denoised = denoise(gray);
    
    // Step 3: Binarize
    cv::Mat binary = binarize(denoised);
    
    // Step 4: Extract edges
    cv::Mat edges = edgeExtract(binary);
    
    return edges;
}

/**
 * @brief Convert image to grayscale
 * 
 * If input image is 3-channel color (BGR format), convert to single-channel grayscale.
 * If input is already grayscale, return a copy to avoid modifying original data.
 * 
 * Using COLOR_BGR2GRAY conversion formula:
 * Gray = 0.299*R + 0.587*G + 0.114*B
 * 
 * @param image Input image (color BGR or grayscale)
 * @return cv::Mat Single-channel grayscale image
 * 
 * 功能：将图像转换为灰度，减少计算量。
 * 如果输入图像是3通道彩色（BGR格式），转换为单通道灰度。
 * 如果输入已经是灰度，则返回副本以避免修改原始数据。
 * 
 * 使用 COLOR_BGR2GRAY 转换公式：
 * Gray = 0.299*R + 0.587*G + 0.114*B
 * 
 * @param image 输入图像（彩色BGR或灰度）
 * @return 单通道灰度图像
 */
cv::Mat ImageProcessor::grayscale(const cv::Mat& image) {
    cv::Mat gray;
    if (image.channels() == 3) {
        // Color image: BGR to grayscale
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        // Already grayscale, return copy
        gray = image.clone();
    }
    return gray;
}

/**
 * @brief Adaptive threshold binarization
 * 
 * Using adaptive Gaussian threshold method, suitable for documents with uneven lighting.
 * Parameters:
 * - maxValue: 255 (white)
 * - adaptiveMethod: ADAPTIVE_THRESH_GAUSSIAN_C (Gaussian weighted average)
 * - thresholdType: THRESH_BINARY_INV (inverted: text white, background black)
 * - blockSize: 11 (neighborhood size, must be odd)
 * - C: 2 (constant subtracted from mean)
 * 
 * Inverted binarization makes text white, better for edge detection.
 * 
 * @param image Input grayscale image
 * @return cv::Mat Binarized image (text white, background black)
 * 
 * 功能：自适应阈值二值化，适合光照不均匀的文档。
 * 使用自适应高斯阈值方法，参数：
 * - maxValue: 255（白色）
 * - adaptiveMethod: ADAPTIVE_THRESH_GAUSSIAN_C（高斯加权平均）
 * - thresholdType: THRESH_BINARY_INV（反转：文本白色，背景黑色）
 * - blockSize: 11（邻域大小，必须为奇数）
 * - C: 2（从平均值中减去的常数）
 * 
 * 反转二值化使文本为白色，更适合边缘检测。
 * 
 * @param image 输入的灰度图像
 * @return 二值化图像（文本白色，背景黑色）
 */
cv::Mat ImageProcessor::binarize(const cv::Mat& image) {
    cv::Mat binary;
    cv::adaptiveThreshold(image, binary, 255, 
                          cv::ADAPTIVE_THRESH_GAUSSIAN_C, 
                          cv::THRESH_BINARY_INV, 11, 2);
    return binary;
}

/**
 * @brief Gaussian blur denoising
 * 
 * Using 5x5 Gaussian kernel for blurring, effectively removes scanning noise and salt-and-pepper noise.
 * Parameters:
 * - ksize: Size(5, 5) (Gaussian kernel size, must be positive odd)
 * - sigmaX: 0 (X direction standard deviation, 0 means auto-calculate)
 * 
 * Gaussian blur preserves edge information while removing noise.
 * 
 * @param image Input image
 * @return cv::Mat Denoised image
 * 
 * 功能：高斯模糊去噪，使用5x5高斯核进行模糊，有效去除扫描噪声和椒盐噪声。
 * 参数：
 * - ksize: Size(5, 5)（高斯核大小，必须为正奇数）
 * - sigmaX: 0（X方向标准差，0表示自动计算）
 * 
 * 高斯模糊在去除噪声的同时保留边缘信息。
 * 
 * @param image 输入图像
 * @return 去噪后的图像
 */
cv::Mat ImageProcessor::denoise(const cv::Mat& image) {
    cv::Mat denoised;
    cv::GaussianBlur(image, denoised, cv::Size(5, 5), 0);
    return denoised;
}

/**
 * @brief Canny edge detection
 * 
 * Using Canny algorithm to detect image edges, crucial step for skew detection.
 * Parameters:
 * - threshold1: 50 (low threshold, for edge linking)
 * - threshold2: 150 (high threshold, for initial edge detection)
 * 
 * Canny algorithm steps:
 * 1. Gaussian filtering to remove noise
 * 2. Calculate gradient magnitude and direction
 * 3. Non-maximum suppression
 * 4. Double threshold detection and edge linking
 * 
 * @param image Input binarized image
 * @return cv::Mat Edge image (edges white, background black)
 * 
 * 功能：Canny边缘检测，使用Canny算法检测图像边缘，是倾斜检测的关键步骤。
 * 参数：
 * - threshold1: 50（低阈值，用于边缘连接）
 * - threshold2: 150（高阈值，用于初始边缘检测）
 * 
 * Canny算法步骤：
 * 1. 高斯滤波去除噪声
 * 2. 计算梯度幅度和方向
 * 3. 非最大值抑制
 * 4. 双阈值检测和边缘连接
 * 
 * @param image 输入的二值化图像
 * @return 边缘图像（边缘白色，背景黑色）
 */
cv::Mat ImageProcessor::edgeExtract(const cv::Mat& image) {
    cv::Mat edges;
    cv::Canny(image, edges, 50, 150);
    return edges;
}
