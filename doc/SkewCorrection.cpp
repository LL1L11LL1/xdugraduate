#include "SkewCorrection.h"

#include <opencv2/opencv.hpp>

using namespace cv;

/**
 * @brief Skew correction implementation
 * 
 * Detailed steps:
 * 1. Calculate rotation center point (image geometric center)
 *    center = (cols/2, rows/2)
 * 
 * 2. Get 2x3 rotation matrix
 *    Using getRotationMatrix2D function with parameters:
 *    - center: rotation center
 *    - angle: rotation angle (negative value means counterclockwise rotation, used for correction)
 *    - scale: 1.0 (maintain original size)
 * 
 * 3. Calculate rotated image size
 *    New size must be large enough to contain the complete rotated image:
 *    newWidth = rows × |sin(θ)| + cols × |cos(θ)|
 *    newHeight = rows × |cos(θ)| + cols × |sin(θ)|
 * 
 * 4. Adjust translation component of rotation matrix
 *    Modify [0,2] and [1,2] elements of rotation matrix to center the rotated image:
 *    rotation[0,2] += (newWidth - cols) / 2
 *    rotation[1,2] += (newHeight - rows) / 2
 * 
 * 5. Perform affine transform
 *    Using warpAffine function with parameters:
 *    - src: input image
 *    - dst: output image
 *    - M: rotation matrix
 *    - dsize: new image size
 *    - flags: INTER_LINEAR (bilinear interpolation)
 *    - borderMode: BORDER_CONSTANT (constant border)
 *    - borderValue: Scalar(255,255,255) (white background)
 * 
 * @param image Input original image
 * @param angle Detected skew angle (degrees)
 * @return Mat Corrected image
 */
Mat SkewCorrection::correctSkew(const Mat& image, double angle) {
    // Get original image size
    int rows = image.rows;
    int cols = image.cols;
    
    // Calculate rotation center (image center point)
    Point2f center(cols / 2.0, rows / 2.0);
    
    // Get rotation matrix (2x3 matrix)
    // Note: angle is negative because the detected angle is the skew angle,
    // need to rotate in reverse direction to correct
    Mat rotation = getRotationMatrix2D(center, angle, 1.0);
    
    // Calculate cos and sin values from rotation matrix (absolute values)
    // rotation.at<double>(0, 0) = cos(θ)
    // rotation.at<double>(0, 1) = -sin(θ)
    double cos = abs(rotation.at<double>(0, 0));
    double sin = abs(rotation.at<double>(0, 1));
    
    // Calculate rotated image size
    // New width = original height × sin(θ) + original width × cos(θ)
    // New height = original height × cos(θ) + original width × sin(θ)
    int newWidth = static_cast<int>((rows * sin) + (cols * cos));
    int newHeight = static_cast<int>((rows * cos) + (cols * sin));
    
    // Adjust translation component of rotation matrix to center the rotated image
    // Modify the 3rd column (translation vector) of rotation matrix
    rotation.at<double>(0, 2) += (newWidth / 2.0) - center.x;
    rotation.at<double>(1, 2) += (newHeight / 2.0) - center.y;
    
    // Perform affine transform (rotation)
    Mat rotated;
    warpAffine(image, rotated, rotation, Size(newWidth, newHeight),
               INTER_LINEAR,           // Bilinear interpolation for better image quality
               BORDER_CONSTANT,        // Constant border filling
               Scalar(255, 255, 255)); // White background (suitable for document images)
    
    return rotated;
}
