/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Implementation of feature extraction functions.
 */

#include <cstdio>
#include <vector>
#include "opencv2/opencv.hpp"
#include "features.h"
#include "faceDetect.h"
#include "filter.h"

/**
 * Extract 5x5 center square as a feature vector.
 */
int baseline5x5(cv::Mat &src, std::vector<float> &feature) {
    if (src.rows < 5 || src.cols < 5) {
        printf("ERROR: Image too small for 5x5 feature extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(75);
    
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;
    int startRow = centerRow - 2;
    int startCol = centerCol - 2;
    
    // Extract pixels in row-major order
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(startRow + i, startCol + j);
            feature.push_back(static_cast<float>(pixel[0]));  // B
            feature.push_back(static_cast<float>(pixel[1]));  // G
            feature.push_back(static_cast<float>(pixel[2]));  // R
        }
    }
    
    return 0;
}

/**
 * Extract 7x7 center square as a feature vector.
 */
int baseline7x7(cv::Mat &src, std::vector<float> &feature) {
    if (src.rows < 7 || src.cols < 7) {
        printf("ERROR: Image too small for 7x7 feature extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(147);
    
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;
    int startRow = centerRow - 3;
    int startCol = centerCol - 3;
    
    // Extract pixels in row-major order
    for (int i = 0; i < 7; i++) {
        for (int j = 0; j < 7; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(startRow + i, startCol + j);
            feature.push_back(static_cast<float>(pixel[0]));  // B
            feature.push_back(static_cast<float>(pixel[1]));  // G
            feature.push_back(static_cast<float>(pixel[2]));  // R
        }
    }
    
    return 0;
}

/**
 * Extract 9x9 center square as a feature vector.
 */
int baseline9x9(cv::Mat &src, std::vector<float> &feature) {
    if (src.rows < 9 || src.cols < 9) {
        printf("ERROR: Image too small for 9x9 feature extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(243);
    
    int centerRow = src.rows / 2;
    int centerCol = src.cols / 2;
    int startRow = centerRow - 4;
    int startCol = centerCol - 4;
    
    // Extract pixels in row-major order
    for (int i = 0; i < 9; i++) {
        for (int j = 0; j < 9; j++) {
            cv::Vec3b pixel = src.at<cv::Vec3b>(startRow + i, startCol + j);
            feature.push_back(static_cast<float>(pixel[0]));  // B
            feature.push_back(static_cast<float>(pixel[1]));  // G
            feature.push_back(static_cast<float>(pixel[2]));  // R
        }
    }
    
    return 0;
}

/**
 * Helper function for RG chromaticity histogram extraction.
 */
static int histogramRG_internal(cv::Mat &src, std::vector<float> &feature, int binsPerChannel) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for histogram extraction\n");
        return -1;
    }
    
    int totalBins = binsPerChannel * binsPerChannel;
    cv::Mat hist = cv::Mat::zeros(binsPerChannel, binsPerChannel, CV_32FC1);
    
    int totalPixels = 0;
    
    // Loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            float B = static_cast<float>(ptr[j][0]);
            float G = static_cast<float>(ptr[j][1]);
            float R = static_cast<float>(ptr[j][2]);
            
            float sum = R + G + B;
            
            // Skip black pixels
            if (sum < 1.0f) continue;
            
            // Calculate chromaticity
            float r = R / sum;
            float g = G / sum;
            
            // Calculate bin indices
            int rBin = static_cast<int>(r * binsPerChannel);
            int gBin = static_cast<int>(g * binsPerChannel);
            
            // Clamp to valid range
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            
            hist.at<float>(rBin, gBin) += 1.0f;
            totalPixels++;
        }
    }
    
    // Normalize by counted pixels
    if (totalPixels > 0) {
        hist /= totalPixels;
    }
    
    // Flatten to vector
    feature.clear();
    feature.reserve(totalBins);
    
    for (int i = 0; i < binsPerChannel; i++) {
        float *histPtr = hist.ptr<float>(i);
        for (int j = 0; j < binsPerChannel; j++) {
            feature.push_back(histPtr[j]);
        }
    }
    
    return 0;
}

/**
 * Extract 2D RG chromaticity histogram with 8 bins per channel.
 */
int histogramRG_8(cv::Mat &src, std::vector<float> &feature) {
    return histogramRG_internal(src, feature, 8);
}

/**
 * Extract 2D RG chromaticity histogram with 16 bins per channel.
 */
int histogramRG_16(cv::Mat &src, std::vector<float> &feature) {
    return histogramRG_internal(src, feature, 16);
}

/**
 * Extract 3D RGB color histogram with 8 bins per channel.
 */
int histogramRGB_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for histogram extraction\n");
        return -1;
    }
    
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    
    cv::Mat hist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    // Loop over all pixels
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            // Calculate bin indices
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            // Clamp to valid range
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            // Calculate 1D index
            int index = rBin * binsPerChannel * binsPerChannel + gBin * binsPerChannel + bBin;
            
            hist.at<float>(index, 0) += 1.0f;
        }
    }
    
    // Normalize by total pixels
    hist /= (src.rows * src.cols);
    
    // Flatten to vector
    feature.clear();
    feature.reserve(totalBins);
    
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(hist.at<float>(i, 0));
    }
    
    return 0;
}

/**
 * Helper function to extract RGB histogram from a specific region.
 */
static int histogramRGB_region(cv::Mat &region, std::vector<float> &feature, int binsPerChannel) {
    if (region.data == NULL) {
        return -1;
    }
    
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat hist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    // Loop over all pixels in the region
    for (int i = 0; i < region.rows; i++) {
        cv::Vec3b *ptr = region.ptr<cv::Vec3b>(i);
        for (int j = 0; j < region.cols; j++) {
            // Calculate bin indices
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            // Clamp to valid range
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            // Calculate 1D index
            int index = rBin * binsPerChannel * binsPerChannel + gBin * binsPerChannel + bBin;
            
            hist.at<float>(index, 0) += 1.0f;
        }
    }
    
    // Normalize by total pixels in region
    hist /= (region.rows * region.cols);
    
    // Append to feature vector
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(hist.at<float>(i, 0));
    }
    
    return 0;
}

/**
 * Extract multi-region RGB histograms (top and bottom halves).
 */
int histogramMultiRGB_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for histogram extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(1024);
    
    int midRow = src.rows / 2;
    
    // Top half
    cv::Mat topHalf = src(cv::Range(0, midRow), cv::Range(0, src.cols));
    if (histogramRGB_region(topHalf, feature, 8) != 0) {
        return -1;
    }
    
    // Bottom half
    cv::Mat bottomHalf = src(cv::Range(midRow, src.rows), cv::Range(0, src.cols));
    if (histogramRGB_region(bottomHalf, feature, 8) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Helper function to compute Sobel magnitude histogram.
 */
static int sobelMagnitudeHistogram(cv::Mat &src, std::vector<float> &feature, int bins) {
    if (src.data == NULL) {
        return -1;
    }
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }
    
    // Use custom Sobel filters from Assignment 1
    cv::Mat sobelX, sobelY, mag;
    sobelX3x3(gray, sobelX);
    sobelY3x3(gray, sobelY);
    magnitude(sobelX, sobelY, mag);
    
    // Create histogram of magnitudes
    cv::Mat hist = cv::Mat::zeros(bins, 1, CV_32FC1);
    
    for (int i = 0; i < mag.rows; i++) {
        unsigned char *ptr = mag.ptr<unsigned char>(i);
        for (int j = 0; j < mag.cols; j++) {
            int bin = ptr[j] / (256 / bins);
            if (bin >= bins) bin = bins - 1;
            hist.at<float>(bin, 0) += 1.0f;
        }
    }
    
    // Normalize
    hist /= (mag.rows * mag.cols);
    
    // Append to feature vector
    for (int i = 0; i < bins; i++) {
        feature.push_back(hist.at<float>(i, 0));
    }
    
    return 0;
}

/**
 * Extract combined texture and color features.
 */
int textureColor_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for texture-color extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(520);
    
    // Extract RGB color histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat colorHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + gBin * binsPerChannel + bBin;
            
            colorHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    colorHist /= (src.rows * src.cols);
    
    // Append color histogram to feature
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(colorHist.at<float>(i, 0));
    }
    
    // Extract texture histogram (8 values)
    if (sobelMagnitudeHistogram(src, feature, 8) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Helper function to compute Gabor filter response histogram.
 */
static int gaborTextureHistogram(cv::Mat &src, std::vector<float> &feature, int bins) {
    if (src.data == NULL) {
        return -1;
    }
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }
    
    gray.convertTo(gray, CV_32F, 1.0 / 255.0);
    
    // Gabor parameters
    int ksize = 31;
    double sigma = 4.0;
    double lambda = 10.0;
    double gamma = 0.5;
    double psi = 0;
    
    std::vector<double> thetas = {0, CV_PI/5, 2*CV_PI/5, 3*CV_PI/5, 4*CV_PI/5};
    
    // Store all histogram values temporarily
    std::vector<float> allHistValues;
    allHistValues.reserve(bins * thetas.size());
    
    for (double theta : thetas) {
        cv::Mat kernel = cv::getGaborKernel(
            cv::Size(ksize, ksize),
            sigma,
            theta,
            lambda,
            gamma,
            psi,
            CV_32F
        );
        
        cv::Mat filtered;
        cv::filter2D(gray, filtered, CV_32F, kernel);
        
        cv::Mat response;
        cv::convertScaleAbs(filtered, response, 255);
        
        // Create histogram (unnormalized for now)
        std::vector<float> hist(bins, 0.0f);
        
        for (int i = 0; i < response.rows; i++) {
            unsigned char *ptr = response.ptr<unsigned char>(i);
            for (int j = 0; j < response.cols; j++) {
                int bin = ptr[j] / (256 / bins);
                if (bin >= bins) bin = bins - 1;
                hist[bin] += 1.0f;
            }
        }
        
        // Add to combined histogram
        for (int i = 0; i < bins; i++) {
            allHistValues.push_back(hist[i]);
        }
    }
    
    // Normalize the entire combined histogram
    float totalCount = 0.0f;
    for (float val : allHistValues) {
        totalCount += val;
    }
    
    if (totalCount > 0.0f) {
        for (float& val : allHistValues) {
            val /= totalCount;
        }
    }
    
    // Append to feature vector
    for (float val : allHistValues) {
        feature.push_back(val);
    }
    
    return 0;
}

/**
 * Extract combined color and Gabor texture features.
 */
int textureColorGabor_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for texture-color-gabor extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(552);
    
    // Extract RGB color histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat colorHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + gBin * binsPerChannel + bBin;
            
            colorHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    colorHist /= (src.rows * src.cols);
    
    // Append color histogram
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(colorHist.at<float>(i, 0));
    }
    
    // Extract Gabor texture histogram (40 values)
    if (gaborTextureHistogram(src, feature, 8) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Helper function to compute Laws filter response histogram.
 */
static int lawsTextureHistogram(cv::Mat &src, std::vector<float> &feature, int bins) {
    if (src.data == NULL) {
        return -1;
    }
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }
    
    gray.convertTo(gray, CV_32F);
    
    // Laws 1D kernels (5-tap)
    float L5[] = {1, 4, 6, 4, 1};
    float E5[] = {-1, -2, 0, 2, 1};
    float S5[] = {-1, 0, 2, 0, -1};
    
    // Create 2D kernels by outer product
    std::vector<std::pair<float*, float*>> kernelPairs = {
        {L5, E5},
        {E5, L5},
        {L5, S5},
        {S5, L5},
        {E5, E5},
        {S5, S5},
        {E5, S5},
        {S5, E5}
    };
    
    // Store all histogram values temporarily
    std::vector<float> allHistValues;
    allHistValues.reserve(bins * kernelPairs.size());
    
    for (auto& pair : kernelPairs) {
        // Create 2D kernel
        cv::Mat kernel(5, 5, CV_32F);
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < 5; j++) {
                kernel.at<float>(i, j) = pair.first[i] * pair.second[j];
            }
        }
        
        // Apply filter
        cv::Mat filtered;
        cv::filter2D(gray, filtered, CV_32F, kernel);
        
        // Take absolute value
        cv::Mat response;
        cv::convertScaleAbs(filtered, response);
        
        // Create histogram (unnormalized for now)
        std::vector<float> hist(bins, 0.0f);
        
        for (int i = 0; i < response.rows; i++) {
            unsigned char *ptr = response.ptr<unsigned char>(i);
            for (int j = 0; j < response.cols; j++) {
                int bin = ptr[j] / (256 / bins);
                if (bin >= bins) bin = bins - 1;
                hist[bin] += 1.0f;
            }
        }
        
        // Add to combined histogram
        for (int i = 0; i < bins; i++) {
            allHistValues.push_back(hist[i]);
        }
    }
    
    // Normalize the entire combined histogram
    float totalCount = 0.0f;
    for (float val : allHistValues) {
        totalCount += val;
    }
    
    if (totalCount > 0.0f) {
        for (float& val : allHistValues) {
            val /= totalCount;
        }
    }
    
    // Append to feature vector
    for (float val : allHistValues) {
        feature.push_back(val);
    }
    
    return 0;
}

/**
 * Extract combined color and Laws texture features.
 */
int textureColorLaws_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for texture-color-laws extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(576);
    
    // Extract RGB color histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat colorHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + 
                       gBin * binsPerChannel + 
                       bBin;
            
            colorHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    colorHist /= (src.rows * src.cols);
    
    // Append color histogram
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(colorHist.at<float>(i, 0));
    }
    
    // Extract Laws texture histogram (64 values)
    if (lawsTextureHistogram(src, feature, 8) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Helper function to compute Fourier power spectrum feature.
 */
static int fourierTextureFeature(cv::Mat &src, std::vector<float> &feature, int targetSize) {
    if (src.data == NULL) {
        return -1;
    }
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }
    
    // Expand to optimal size for DFT
    cv::Mat padded;
    int m = cv::getOptimalDFTSize(gray.rows);
    int n = cv::getOptimalDFTSize(gray.cols);
    cv::copyMakeBorder(gray, padded, 0, m - gray.rows, 0, n - gray.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Convert to float
    padded.convertTo(padded, CV_32F);
    
    // Create complex image
    cv::Mat planes[] = {padded, cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexImg;
    cv::merge(planes, 2, complexImg);
    
    // Compute DFT
    cv::dft(complexImg, complexImg);
    
    // Compute magnitude (power spectrum)
    cv::split(complexImg, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magnitude = planes[0];
    
    // Log transform for better visualization
    magnitude += cv::Scalar::all(1);
    cv::log(magnitude, magnitude);
    
    // Shift quadrants (move DC to center)
    int cx = magnitude.cols / 2;
    int cy = magnitude.rows / 2;
    cv::Mat q0(magnitude, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magnitude, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magnitude, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magnitude, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    // Resize to target size (16x16)
    cv::Mat resized;
    cv::resize(magnitude, resized, cv::Size(targetSize, targetSize));
    
    // Normalize to [0, 1]
    cv::normalize(resized, resized, 0, 1, cv::NORM_MINMAX);
    
    // Flatten to feature vector
    for (int i = 0; i < resized.rows; i++) {
        float *ptr = resized.ptr<float>(i);
        for (int j = 0; j < resized.cols; j++) {
            feature.push_back(ptr[j]);
        }
    }
    
    return 0;
}

/**
 * Extract combined color and Fourier texture features.
 */
int textureColorFourier_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for texture-color-fourier extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(768);
    
    // Extract RGB color histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat colorHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + 
                       gBin * binsPerChannel + 
                       bBin;
            
            colorHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    colorHist /= (src.rows * src.cols);
    
    // Append color histogram
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(colorHist.at<float>(i, 0));
    }
    
    // Extract Fourier texture feature (256 values)
    if (fourierTextureFeature(src, feature, 16) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Helper function to compute CM features.
 */
static int cmTextureFeatures(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        return -1;
    }
    
    cv::Mat gray;
    if (src.channels() == 3) {
        cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = src;
    }
    
    // Quantize to 8 levels for faster computation
    const int levels = 8;
    cv::Mat quantized;
    gray.convertTo(quantized, CV_8U);
    quantized = quantized / (256 / levels);
    
    // Define 4 directions: (dx, dy)
    std::vector<std::pair<int, int>> directions = {
        {1, 0},   // 0°   - horizontal
        {1, 1},   // 45°  - diagonal
        {0, 1},   // 90°  - vertical
        {-1, 1}   // 135° - anti-diagonal
    };
    
    for (auto& dir : directions) {
        int dx = dir.first;
        int dy = dir.second;
        
        // Build CM
        cv::Mat cm = cv::Mat::zeros(levels, levels, CV_32F);
        
        for (int i = 0; i < quantized.rows; i++) {
            for (int j = 0; j < quantized.cols; j++) {
                int ni = i + dy;
                int nj = j + dx;
                
                // Check bounds
                if (ni >= 0 && ni < quantized.rows && nj >= 0 && nj < quantized.cols) {
                    int val1 = quantized.at<uchar>(i, j);
                    int val2 = quantized.at<uchar>(ni, nj);
                    
                    if (val1 >= levels) val1 = levels - 1;
                    if (val2 >= levels) val2 = levels - 1;
                    
                    cm.at<float>(val1, val2) += 1.0f;
                }
            }
        }
        
        // Normalize CM
        float sum = cv::sum(cm)[0];
        if (sum > 0) {
            cm /= sum;
        }
        
        // Compute features
        float energy = 0.0f;
        float entropy = 0.0f;
        float contrast = 0.0f;
        float homogeneity = 0.0f;
        float correlation = 0.0f;
        
        // Mean values for correlation
        float meanI = 0.0f, meanJ = 0.0f;
        for (int i = 0; i < levels; i++) {
            for (int j = 0; j < levels; j++) {
                float p = cm.at<float>(i, j);
                meanI += i * p;
                meanJ += j * p;
            }
        }
        
        // Standard deviations for correlation
        float stdI = 0.0f, stdJ = 0.0f;
        for (int i = 0; i < levels; i++) {
            for (int j = 0; j < levels; j++) {
                float p = cm.at<float>(i, j);
                stdI += p * (i - meanI) * (i - meanI);
                stdJ += p * (j - meanJ) * (j - meanJ);
            }
        }
        stdI = std::sqrt(stdI);
        stdJ = std::sqrt(stdJ);
        
        // Calculate all features
        for (int i = 0; i < levels; i++) {
            for (int j = 0; j < levels; j++) {
                float p = cm.at<float>(i, j);
                
                if (p > 0) {
                    // Energy (Angular Second Moment)
                    energy += p * p;
                    
                    // Entropy
                    entropy -= p * std::log(p + 1e-10f);
                    
                    // Contrast
                    contrast += (i - j) * (i - j) * p;
                    
                    // Homogeneity (Inverse Difference Moment)
                    homogeneity += p / (1.0f + (i - j) * (i - j));
                    
                    // Correlation
                    if (stdI > 0 && stdJ > 0) {
                        correlation += ((i - meanI) * (j - meanJ) * p) / (stdI * stdJ);
                    }
                }
            }
        }
        
        // Append features for this direction
        feature.push_back(energy);
        feature.push_back(entropy);
        feature.push_back(contrast);
        feature.push_back(homogeneity);
        feature.push_back(correlation);
    }
    
    return 0;
}

/**
 * Extract combined color and CM texture features.
 */
int textureColorCM_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for texture-color-cm extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(532);
    
    // Extract RGB color histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat colorHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + 
                       gBin * binsPerChannel + 
                       bBin;
            
            colorHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    colorHist /= (src.rows * src.cols);
    
    // Append color histogram
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(colorHist.at<float>(i, 0));
    }
    
    // Extract CM texture features (20 values)
    if (cmTextureFeatures(src, feature) != 0) {
        return -1;
    }
    
    return 0;
}

/**
 * Extract centered salient object features (generic, color-agnostic).
 */
int customCenteredObjectFeatures(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for centered object feature extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(531);
    
    int centerX = src.cols / 2;
    int centerY = src.rows / 2;
    int radius = std::min(src.rows, src.cols) / 5;  // Center 40% circle
    
    // Component 1: Center region RGB histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat centerHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    int centerPixels = 0;
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int dx = j - centerX;
            int dy = i - centerY;
            
            // Only center circle
            if (dx*dx + dy*dy <= radius*radius) {
                int bBin = ptr[j][0] / (256 / binsPerChannel);
                int gBin = ptr[j][1] / (256 / binsPerChannel);
                int rBin = ptr[j][2] / (256 / binsPerChannel);
                
                if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
                if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
                if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
                
                int index = rBin * binsPerChannel * binsPerChannel + 
                           gBin * binsPerChannel + 
                           bBin;
                
                centerHist.at<float>(index, 0) += 1.0f;
                centerPixels++;
            }
        }
    }
    
    centerHist /= centerPixels;
    
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(centerHist.at<float>(i, 0));
    }
    
    // Component 2: Center vs Periphery color difference (3 values)
    float centerR = 0, centerG = 0, centerB = 0;
    float peripheryR = 0, peripheryG = 0, peripheryB = 0;
    int peripheryPixels = 0;
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int dx = j - centerX;
            int dy = i - centerY;
            
            if (dx*dx + dy*dy <= radius*radius) {
                centerR += ptr[j][2];
                centerG += ptr[j][1];
                centerB += ptr[j][0];
            } else {
                peripheryR += ptr[j][2];
                peripheryG += ptr[j][1];
                peripheryB += ptr[j][0];
                peripheryPixels++;
            }
        }
    }
    
    if (centerPixels > 0) {
        centerR /= centerPixels;
        centerG /= centerPixels;
        centerB /= centerPixels;
    }
    if (peripheryPixels > 0) {
        peripheryR /= peripheryPixels;
        peripheryG /= peripheryPixels;
        peripheryB /= peripheryPixels;
    }
    
    float colorDiffR = std::abs(centerR - peripheryR) / 255.0f;
    float colorDiffG = std::abs(centerG - peripheryG) / 255.0f;
    float colorDiffB = std::abs(centerB - peripheryB) / 255.0f;
    
    feature.push_back(colorDiffR);
    feature.push_back(colorDiffG);
    feature.push_back(colorDiffB);
    
    // Component 3: Saturation concentration in center (1 value)
    cv::Mat hsv;
    cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
    
    float avgSaturation = 0.0f;
    int satCount = 0;
    
    for (int i = 0; i < hsv.rows; i++) {
        cv::Vec3b *ptr = hsv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hsv.cols; j++) {
            int dx = j - centerX;
            int dy = i - centerY;
            
            if (dx*dx + dy*dy <= radius*radius) {
                avgSaturation += ptr[j][1];  // S channel
                satCount++;
            }
        }
    }
    
    if (satCount > 0) {
        avgSaturation /= (satCount * 255.0f);
    }
    feature.push_back(avgSaturation);
    
    // Component 4: Saturation histogram in center (8 values)
    std::vector<float> satHist(8, 0.0f);
    
    for (int i = 0; i < hsv.rows; i++) {
        cv::Vec3b *ptr = hsv.ptr<cv::Vec3b>(i);
        for (int j = 0; j < hsv.cols; j++) {
            int dx = j - centerX;
            int dy = i - centerY;
            
            if (dx*dx + dy*dy <= radius*radius) {
                int bin = ptr[j][1] / (256 / 8);  // S channel
                if (bin >= 8) bin = 7;
                satHist[bin] += 1.0f;
            }
        }
    }
    
    if (centerPixels > 0) {
        for (float& val : satHist) val /= centerPixels;
    }
    
    for (float val : satHist) {
        feature.push_back(val);
    }
    
    // Component 5: Background color statistics (6 values)
    float bgMeanR = peripheryR;
    float bgMeanG = peripheryG;
    float bgMeanB = peripheryB;
    
    float bgStdR = 0, bgStdG = 0, bgStdB = 0;
    
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            int dx = j - centerX;
            int dy = i - centerY;
            
            if (dx*dx + dy*dy > radius*radius) {
                bgStdR += (ptr[j][2] - bgMeanR) * (ptr[j][2] - bgMeanR);
                bgStdG += (ptr[j][1] - bgMeanG) * (ptr[j][1] - bgMeanG);
                bgStdB += (ptr[j][0] - bgMeanB) * (ptr[j][0] - bgMeanB);
            }
        }
    }
    
    if (peripheryPixels > 0) {
        bgStdR = std::sqrt(bgStdR / peripheryPixels) / 255.0f;
        bgStdG = std::sqrt(bgStdG / peripheryPixels) / 255.0f;
        bgStdB = std::sqrt(bgStdB / peripheryPixels) / 255.0f;
    }
    
    feature.push_back(bgMeanR / 255.0f);
    feature.push_back(bgMeanG / 255.0f);
    feature.push_back(bgMeanB / 255.0f);
    feature.push_back(bgStdR);
    feature.push_back(bgStdG);
    feature.push_back(bgStdB);
    
    return 0;
}

/**
 * Extract blue sky outdoor scene features.
 */
int customBlueSkyFeatures(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for blue sky feature extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(547);
    
    // Define top third of image
    int topHeight = src.rows / 3;
    cv::Mat topRegion = src(cv::Range(0, topHeight), cv::Range(0, src.cols));
    
    // Component 1: Top-third RGB histogram (512 values)
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    cv::Mat topHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
    
    for (int i = 0; i < topRegion.rows; i++) {
        cv::Vec3b *ptr = topRegion.ptr<cv::Vec3b>(i);
        for (int j = 0; j < topRegion.cols; j++) {
            int bBin = ptr[j][0] / (256 / binsPerChannel);
            int gBin = ptr[j][1] / (256 / binsPerChannel);
            int rBin = ptr[j][2] / (256 / binsPerChannel);
            
            if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            
            int index = rBin * binsPerChannel * binsPerChannel + 
                       gBin * binsPerChannel + 
                       bBin;
            
            topHist.at<float>(index, 0) += 1.0f;
        }
    }
    
    topHist /= (topRegion.rows * topRegion.cols);
    
    for (int i = 0; i < totalBins; i++) {
        feature.push_back(topHist.at<float>(i, 0));
    }
    
    // Component 2: Sky blueness score (1 value)
    float blueness = 0.0f;
    for (int i = 0; i < topRegion.rows; i++) {
        cv::Vec3b *ptr = topRegion.ptr<cv::Vec3b>(i);
        for (int j = 0; j < topRegion.cols; j++) {
            float B = ptr[j][0];
            float G = ptr[j][1];
            float R = ptr[j][2];
            
            // Blue dominance: B - (R+G)/2
            blueness += (B - (R + G) / 2.0f);
        }
    }
    blueness /= (topRegion.rows * topRegion.cols * 255.0f);  // Normalize to [0, 1]
    feature.push_back(blueness);
    
    // Component 3: Vertical edge histogram (8 values)
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    
    cv::Mat sobelX, sobelY;
    cv::Sobel(gray, sobelX, CV_16S, 1, 0, 3);
    cv::Sobel(gray, sobelY, CV_16S, 0, 1, 3);
    
    cv::Mat absX, absY;
    cv::convertScaleAbs(sobelX, absX);
    cv::convertScaleAbs(sobelY, absY);
    
    std::vector<float> edgeHist(8, 0.0f);
    
    for (int i = 0; i < gray.rows; i++) {
        for (int j = 0; j < gray.cols; j++) {
            float gx = absX.at<uchar>(i, j);
            float gy = absY.at<uchar>(i, j);
            
            // Vertical edge strength (gx > gy means vertical)
            if (gx > gy && gx > 20) {
                int bin = static_cast<int>(gx / (256.0f / 8));
                if (bin >= 8) bin = 7;
                edgeHist[bin] += 1.0f;
            }
        }
    }
    
    // Normalize
    float edgeSum = 0.0f;
    for (float val : edgeHist) edgeSum += val;
    if (edgeSum > 0) {
        for (float& val : edgeHist) val /= edgeSum;
    }
    
    for (float val : edgeHist) {
        feature.push_back(val);
    }
    
    // Component 4: Brightness gradient (top to bottom) (1 value)
    float topBrightness = 0.0f;
    float bottomBrightness = 0.0f;
    int bottomHeight = src.rows / 3;
    
    for (int i = 0; i < topHeight; i++) {
        uchar *ptr = gray.ptr<uchar>(i);
        for (int j = 0; j < gray.cols; j++) {
            topBrightness += ptr[j];
        }
    }
    topBrightness /= (topHeight * gray.cols);
    
    for (int i = src.rows - bottomHeight; i < src.rows; i++) {
        uchar *ptr = gray.ptr<uchar>(i);
        for (int j = 0; j < gray.cols; j++) {
            bottomBrightness += ptr[j];
        }
    }
    bottomBrightness /= (bottomHeight * gray.cols);
    
    float gradient = (topBrightness - bottomBrightness) / 255.0f;
    feature.push_back(gradient);
    
    // Component 5: Texture smoothness in top region (1 value)
    cv::Mat topGray = gray(cv::Range(0, topHeight), cv::Range(0, gray.cols));
    cv::Scalar mean, stddev;
    cv::meanStdDev(topGray, mean, stddev);
    float smoothness = stddev[0] / 255.0f;  // Lower = smoother (sky)
    feature.push_back(smoothness);
    
    // Component 6: Blue channel dominance histogram in top (8 values)
    std::vector<float> blueHist(8, 0.0f);
    
    for (int i = 0; i < topRegion.rows; i++) {
        cv::Vec3b *ptr = topRegion.ptr<cv::Vec3b>(i);
        for (int j = 0; j < topRegion.cols; j++) {
            float B = ptr[j][0];
            float R = ptr[j][2];
            float G = ptr[j][1];
            float sum = R + G + B;
            
            if (sum > 10) {
                float blueDom = B / sum;
                int bin = static_cast<int>(blueDom * 8);
                if (bin >= 8) bin = 7;
                blueHist[bin] += 1.0f;
            }
        }
    }
    
    // Normalize
    float blueSum = 0.0f;
    for (float val : blueHist) blueSum += val;
    if (blueSum > 0) {
        for (float& val : blueHist) val /= blueSum;
    }
    
    for (float val : blueHist) {
        feature.push_back(val);
    }
    
    // Component 7: Cloud texture (high-frequency in top region) (16 values)
    cv::Mat topBlurred;
    cv::GaussianBlur(topGray, topBlurred, cv::Size(5, 5), 1.0);
    cv::Mat highFreq = topGray - topBlurred;
    
    std::vector<float> cloudHist(16, 0.0f);
    
    for (int i = 0; i < highFreq.rows; i++) {
        char *ptr = highFreq.ptr<char>(i);  // Can be negative
        for (int j = 0; j < highFreq.cols; j++) {
            int val = std::abs(ptr[j]);
            int bin = val / (256 / 16);
            if (bin >= 16) bin = 15;
            cloudHist[bin] += 1.0f;
        }
    }
    
    // Normalize
    float cloudSum = 0.0f;
    for (float val : cloudHist) cloudSum += val;
    if (cloudSum > 0) {
        for (float& val : cloudHist) val /= cloudSum;
    }
    
    for (float val : cloudHist) {
        feature.push_back(val);
    }
    
    return 0;
}

/**
 * Extract face-aware RGB histograms.
 */
int faceAwareRGB_8(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for face-aware extraction\n");
        return -1;
    }
    
    feature.clear();
    feature.reserve(1024);
    
    const int binsPerChannel = 8;
    int totalBins = binsPerChannel * binsPerChannel * binsPerChannel;
    
    // Detect faces
    cv::Mat gray;
    cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> faces;
    detectFaces(gray, faces);
    
    if (faces.size() > 0) {
        // Face detected - extract face and background histograms
        
        // Create mask for face region
        cv::Mat faceMask = cv::Mat::zeros(src.size(), CV_8U);
        for (const auto& face : faces) {
            cv::rectangle(faceMask, face, cv::Scalar(255), -1);  // Fill face regions
        }
        
        // Face histogram
        cv::Mat faceHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
        int facePixels = 0;
        
        for (int i = 0; i < src.rows; i++) {
            cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
            unsigned char *maskPtr = faceMask.ptr<unsigned char>(i);
            for (int j = 0; j < src.cols; j++) {
                if (maskPtr[j] > 0) {  // Inside face region
                    int bBin = ptr[j][0] / (256 / binsPerChannel);
                    int gBin = ptr[j][1] / (256 / binsPerChannel);
                    int rBin = ptr[j][2] / (256 / binsPerChannel);
                    
                    if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
                    if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
                    if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
                    
                    int index = rBin * binsPerChannel * binsPerChannel + 
                               gBin * binsPerChannel + 
                               bBin;
                    
                    faceHist.at<float>(index, 0) += 1.0f;
                    facePixels++;
                }
            }
        }
        
        if (facePixels > 0) {
            faceHist /= facePixels;
        }
        
        // Background histogram
        cv::Mat bgHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
        int bgPixels = 0;
        
        for (int i = 0; i < src.rows; i++) {
            cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
            unsigned char *maskPtr = faceMask.ptr<unsigned char>(i);
            for (int j = 0; j < src.cols; j++) {
                if (maskPtr[j] == 0) {  // Outside face region
                    int bBin = ptr[j][0] / (256 / binsPerChannel);
                    int gBin = ptr[j][1] / (256 / binsPerChannel);
                    int rBin = ptr[j][2] / (256 / binsPerChannel);
                    
                    if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
                    if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
                    if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
                    
                    int index = rBin * binsPerChannel * binsPerChannel + 
                               gBin * binsPerChannel + 
                               bBin;
                    
                    bgHist.at<float>(index, 0) += 1.0f;
                    bgPixels++;
                }
            }
        }
        
        if (bgPixels > 0) {
            bgHist /= bgPixels;
        }
        
        // Append face histogram
        for (int i = 0; i < totalBins; i++) {
            feature.push_back(faceHist.at<float>(i, 0));
        }
        
        // Append background histogram
        for (int i = 0; i < totalBins; i++) {
            feature.push_back(bgHist.at<float>(i, 0));
        }
        
    } else {
        // No face detected - use whole image histogram + zeros
        
        cv::Mat wholeHist = cv::Mat::zeros(totalBins, 1, CV_32FC1);
        
        for (int i = 0; i < src.rows; i++) {
            cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
            for (int j = 0; j < src.cols; j++) {
                int bBin = ptr[j][0] / (256 / binsPerChannel);
                int gBin = ptr[j][1] / (256 / binsPerChannel);
                int rBin = ptr[j][2] / (256 / binsPerChannel);
                
                if (bBin >= binsPerChannel) bBin = binsPerChannel - 1;
                if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
                if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
                
                int index = rBin * binsPerChannel * binsPerChannel + 
                           gBin * binsPerChannel + 
                           bBin;
                
                wholeHist.at<float>(index, 0) += 1.0f;
            }
        }
        
        wholeHist /= (src.rows * src.cols);
        
        // Append whole image histogram
        for (int i = 0; i < totalBins; i++) {
            feature.push_back(wholeHist.at<float>(i, 0));
        }
        
        // Append zeros for background (no face detected)
        for (int i = 0; i < totalBins; i++) {
            feature.push_back(0.0f);
        }
    }
    
    return 0;
}

/**
 * Extract 2D RG chromaticity histogram with Gaussian smoothing.
 */
int histogramRG_16_smooth(cv::Mat &src, std::vector<float> &feature) {
    if (src.data == NULL) {
        printf("ERROR: Invalid image for histogram extraction\n");
        return -1;
    }
    
    const int binsPerChannel = 16;
    int totalBins = binsPerChannel * binsPerChannel;
    cv::Mat hist = cv::Mat::zeros(binsPerChannel, binsPerChannel, CV_32FC1);
    
    // Build histogram
    for (int i = 0; i < src.rows; i++) {
        cv::Vec3b *ptr = src.ptr<cv::Vec3b>(i);
        for (int j = 0; j < src.cols; j++) {
            float B = static_cast<float>(ptr[j][0]);
            float G = static_cast<float>(ptr[j][1]);
            float R = static_cast<float>(ptr[j][2]);
            
            float sum = R + G + B;
            if (sum < 1.0f) continue;
            
            float r = R / sum;
            float g = G / sum;
            
            int rBin = static_cast<int>(r * binsPerChannel);
            int gBin = static_cast<int>(g * binsPerChannel);
            
            if (rBin >= binsPerChannel) rBin = binsPerChannel - 1;
            if (gBin >= binsPerChannel) gBin = binsPerChannel - 1;
            
            hist.at<float>(rBin, gBin) += 1.0f;
        }
    }
    
    // Normalize
    hist /= (src.rows * src.cols);
    
    cv::Mat smoothedHist;
    blur5x5_2(hist, smoothedHist);
    
    // Re-normalize after smoothing
    float totalSum = cv::sum(smoothedHist)[0];
    if (totalSum > 0) {
        smoothedHist /= totalSum;
    }
    
    // Flatten to vector
    feature.clear();
    feature.reserve(totalBins);
    
    for (int i = 0; i < binsPerChannel; i++) {
        float *histPtr = smoothedHist.ptr<float>(i);
        for (int j = 0; j < binsPerChannel; j++) {
            feature.push_back(histPtr[j]);
        }
    }
    
    return 0;
}