#include "PerformanceEvaluator.h"
#include "ImageProcessor.h"
#include "SkewDetection.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <chrono>
#include <iostream>

using namespace cv;
using namespace std;
using namespace chrono;

/**
 * @brief Batch evaluate test set implementation
 * 
 * Detailed steps:
 * 1. Use cv::glob to get all image files in test directory (.jpg and .png formats)
 * 2. Iterate through all images, call evaluateSingleImage for evaluation
 * 3. Accumulate angle errors and processing times for all images
 * 4. Calculate averages and output results
 * 
 * Supported algorithms:
 * - "hough": Hough transform algorithm
 * - "projection": Projection analysis algorithm
 * - "leastSquares": Least squares algorithm
 * 
 * Output information:
 * - Algorithm name
 * - Average angle error (degrees)
 * - Average processing time (milliseconds)
 * 
 * @param testDir Test image directory path
 * @param algorithmName Algorithm name string
 */
void PerformanceEvaluator::evaluate(const string& testDir, const string& algorithmName) {
    // Store all image paths
    vector<cv::String> imagePaths;
    
    // Get .jpg format images
    glob(testDir + "/*.jpg", imagePaths, false);
    // Get .png format images (append to list)
    glob(testDir + "/*.png", imagePaths, true);
    
    // Accumulation variables
    double totalError = 0;
    double totalTime = 0;
    
    // Iterate through all test images
    for (const cv::String& path : imagePaths) {
        // Read image
        Mat image = imread(path);
        if (image.empty()) continue;  // Skip unreadable images
        
        // Select algorithm based on algorithm name and evaluate
        Result result;
        if (algorithmName == "hough") {
            result = evaluateSingleImage(image, SkewDetection::HOUGH_TRANSFORM);
        } else if (algorithmName == "projection") {
            result = evaluateSingleImage(image, SkewDetection::PROJECTION_ANALYSIS);
        } else if (algorithmName == "leastSquares") {
            result = evaluateSingleImage(image, SkewDetection::LEAST_SQUARES);
        }
        
        // Accumulate results
        totalError += result.angleError;
        totalTime += result.processingTime;
    }
    
    // Calculate averages and output
    if (imagePaths.size() > 0) {
        cout << "=================================" << endl;
        cout << "Algorithm: " << algorithmName << endl;
        cout << "Number of test images: " << imagePaths.size() << endl;
        cout << "Average angle error: " << totalError / imagePaths.size() << " degrees" << endl;
        cout << "Average processing time: " << totalTime / imagePaths.size() << " milliseconds" << endl;
        cout << "=================================" << endl;
    } else {
        cout << "No test images found" << endl;
    }
}

/**
 * @brief Single image evaluation implementation
 * 
 * Detailed steps:
 * 1. Record start time
 * 2. Perform image preprocessing (grayscale, denoise, binarize, edge extract)
 * 3. Perform skew detection
 * 4. Record end time
 * 5. Calculate processing time
 * 6. Calculate angle error (assuming true angle is 0 degrees)
 * 
 * Note: Current implementation assumes true skew angle of test images is 0 degrees,
 * so angle error equals absolute value of detected angle.
 * In practical applications, test sets with known skew angles can be used for evaluation.
 * 
 * @param image Input original image
 * @param algorithm Algorithm type enum value
 * @return Result Evaluation result structure
 */
PerformanceEvaluator::Result PerformanceEvaluator::evaluateSingleImage(const Mat& image, int algorithm) {
    Result result;
    
    // Record start time
    auto start = high_resolution_clock::now();
    
    // Perform image preprocessing
    Mat processed = ImageProcessor::preprocess(image);
    
    // Perform skew detection
    double detectedAngle = SkewDetection::detectSkew(processed, 
        static_cast<SkewDetection::Algorithm>(algorithm));
    
    // Record end time
    auto end = high_resolution_clock::now();
    
    // Calculate processing time (milliseconds)
    result.processingTime = duration<double, milli>(end - start).count();
    
    // Calculate angle error (assuming true angle is 0 degrees)
    // In practical applications, should compare with true angle
    result.angleError = abs(detectedAngle);
    
    return result;
}
