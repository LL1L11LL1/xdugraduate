#include "SkewDetection.h"

#include <vector>
#include <algorithm>
#include <cmath>

using namespace cv;
using namespace std;

/**
 * @brief Unified skew detection interface implementation
 * 
 * Based on user-selected algorithm type, call corresponding detection method.
 * Provides unified interface for easy algorithm switching in comparison.
 * 
 * @param image Preprocessed edge image
 * @param algorithm Algorithm type enum value
 * @return double Detected skew angle (degrees)
 */
double SkewDetection::detectSkew(const Mat& image, Algorithm algorithm) {
    switch (algorithm) {
    case HOUGH_TRANSFORM:
        // Hough transform: suitable for complex layouts, high precision
        return houghTransform(image);
    case PROJECTION_ANALYSIS:
        // Projection analysis: suitable for simple text
        return projectionAnalysis(image);
    case LEAST_SQUARES:
        // Least squares: fast computation
        return leastSquares(image);
    case RADON_TRANSFORM:
        // Radon transform: robust to noise
        return radonTransform(image);
    default:
        // Default return 0 degrees (no skew)
        return 0.0;
    }
}

/**
 * @brief Hough transform skew detection implementation
 * 
 * Detailed steps:
 * 1. Use HoughLines function to detect lines in edge image
 *    - rho resolution: 1 pixel
 *    - theta resolution: 1 degree (CV_PI/180 radians)
 *    - threshold: 100 (at least 100 points on line)
 * 2. Extract angle theta for each line, convert to degrees
 * 3. Filter valid angles in -45° to 45° range
 * 4. Sort valid angles and take median for robustness
 * 
 * Mathematical principle:
 * Hough transform converts lines in image space to points in parameter space (rho, theta).
 * Where rho is distance from line to origin, theta is line angle.
 * 
 * @param image Edge image
 * @return double Skew angle (degrees), positive value means clockwise skew
 */
double SkewDetection::houghTransform(const Mat& image) {
    // Store detected line parameters (rho, theta)
    vector<Vec2f> lines;
    
    // Hough transform to detect lines
    // Parameters: image, output lines, rho resolution, theta resolution, threshold
    HoughLines(image, lines, 1, CV_PI / 180, 100);

    // Store all valid angles
    vector<double> angles;
    
    // Iterate through all detected lines
    for (size_t i = 0; i < lines.size(); i++) {
        // Extract theta parameter (radians)
        double theta = lines[i][1];
        
        // Convert to degrees and adjust range
        // theta range is 0 to 180 degrees, subtract 90 degrees to get -90 to 90 degrees
        double angle = theta * 180 / CV_PI - 90;
        
        // Only keep angles in -45° to 45° range (normal document skew range)
        if (abs(angle) < 45) {
            angles.push_back(angle);
        }
    }

    // If no valid lines detected, return 0 degrees
    if (angles.empty()) {
        return 0.0;
    }

    // Sort angles and take median for robustness
    sort(angles.begin(), angles.end());
    return angles[angles.size() / 2];
}

/**
 * @brief Projection analysis skew detection implementation
 * 
 * Detailed steps:
 * 1. In -45° to 45° range, iterate through all angles with 0.5° step
 * 2. For each angle, use affine transform to rotate image
 * 3. Calculate vertical projection energy of rotated image
 *    - Projection energy = sum of squares of white pixel count per column
 * 4. Angle with maximum projection energy is the skew angle
 * 
 * Principle explanation:
 * When image is rotated to correct angle, text lines are horizontally aligned,
 * vertical projection shows obvious peaks (text line positions),
 * projection energy reaches maximum.
 * 
 * @param image Edge image or binary image
 * @return double Skew angle (degrees)
 */
double SkewDetection::projectionAnalysis(const Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    
    // Record best angle and maximum projection energy
    double bestAngle = 0.0;
    double maxProjection = 0.0;
    
    // Search in -45° to 45° range
    for (double angle = -45; angle <= 45; angle += 0.5) {
        Mat rotated;
        
        // Calculate rotation center
        Point2f center(cols / 2.0, rows / 2.0);
        
        // Get rotation matrix
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);
        
        // Perform affine transform (rotation)
        warpAffine(image, rotated, rotation, Size(cols, rows));
        
        // Calculate vertical projection energy
        double projection = 0;
        for (int x = 0; x < rotated.cols; x++) {
            int count = 0;
            // Count white pixels in current column
            for (int y = 0; y < rotated.rows; y++) {
                if (rotated.at<uchar>(y, x) > 0) {
                    count++;
                }
            }
            // Accumulate squared values (enhance peaks)
            projection += count * count;
        }
        
        // Update best angle
        if (projection > maxProjection) {
            maxProjection = projection;
            bestAngle = angle;
        }
    }
    
    return bestAngle;
}

/**
 * @brief Least squares skew detection implementation
 * 
 * Detailed steps:
 * 1. Iterate through edge image, extract coordinates of all white pixels
 * 2. Use least squares method to fit line y = kx + b
 *    - Calculate necessary statistics: sumX, sumY, sumXY, sumX2
 * 3. Calculate skew angle from slope k: angle = arctan(k) * 180/π
 * 
 * Least squares formula:
 * Slope k = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
 * 
 * @param image Edge image
 * @return double Skew angle (degrees)
 */
double SkewDetection::leastSquares(const Mat& image) {
    // Store all edge point coordinates
    vector<Point> points;
    
    // Iterate through image, extract white pixel points
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            if (image.at<uchar>(y, x) > 0) {
                points.push_back(Point(x, y));
            }
        }
    }
    
    // If too few points to fit line, return 0 degrees
    if (points.size() < 2) {
        return 0.0;
    }
    
    // Least squares calculation
    double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    
    // Accumulate statistics
    for (const Point& p : points) {
        sumX += p.x;
        sumY += p.y;
        sumXY += p.x * p.y;
        sumX2 += p.x * p.x;
    }
    
    int n = points.size();
    
    // Calculate slope k = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
    double denominator = n * sumX2 - sumX * sumX;
    
    // Prevent division by zero
    if (abs(denominator) < 1e-10) {
        return 0.0;
    }
    
    double slope = (n * sumXY - sumX * sumY) / denominator;
    
    // Calculate angle from slope: angle = arctan(slope) * 180/π
    double angle = atan(slope) * 180 / CV_PI;
    
    return angle;
}

/**
 * @brief Radon transform skew detection implementation
 * 
 * Detailed steps:
 * 1. Create a Radon transform buffer with angles from -45° to 45°
 * 2. For each angle in this range, simulate the Radon transform by rotating the image
 *    and calculating the sum of pixel values along each row
 * 3. Find the angle that produces the maximum sum (indicating the dominant line direction)
 * 4. Convert this angle to the skew angle
 * 
 * Principle explanation:
 * The Radon transform projects an image onto a series of lines at different angles.
 * The angle with the strongest projection (maximum sum) corresponds to the dominant
 * orientation in the image, which is the skew angle of the document.
 * 
 * @param image Edge image
 * @return double Skew angle (degrees)
 */
double SkewDetection::radonTransform(const Mat& image) {
    int rows = image.rows;
    int cols = image.cols;
    
    // Define angle range and step
    const double minAngle = -45.0;
    const double maxAngle = 45.0;
    const double angleStep = 0.5;
    int numAngles = static_cast<int>((maxAngle - minAngle) / angleStep) + 1;
    
    // Store maximum projection value for each angle
    vector<double> projectionMax(numAngles, 0.0);
    
    // Iterate through all angles
    for (int i = 0; i < numAngles; i++) {
        double angle = minAngle + i * angleStep;
        Mat rotated;
        
        // Calculate rotation center
        Point2f center(cols / 2.0, rows / 2.0);
        
        // Get rotation matrix
        Mat rotation = getRotationMatrix2D(center, angle, 1.0);
        
        // Perform affine transform (rotation)
        warpAffine(image, rotated, rotation, Size(cols, rows));
        
        // Calculate horizontal projection (sum of each row)
        double maxSum = 0.0;
        for (int y = 0; y < rotated.rows; y++) {
            double rowSum = 0.0;
            for (int x = 0; x < rotated.cols; x++) {
                rowSum += rotated.at<uchar>(y, x);
            }
            if (rowSum > maxSum) {
                maxSum = rowSum;
            }
        }
        
        projectionMax[i] = maxSum;
    }
    
    // Find angle with maximum projection
    int bestAngleIndex = 0;
    double maxProjection = projectionMax[0];
    for (int i = 1; i < numAngles; i++) {
        if (projectionMax[i] > maxProjection) {
            maxProjection = projectionMax[i];
            bestAngleIndex = i;
        }
    }
    
    // Convert index to angle
    double bestAngle = minAngle + bestAngleIndex * angleStep;
    
    return bestAngle;
}
