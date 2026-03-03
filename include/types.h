/**
 * Rohil Kulshreshtha
 * January 31, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Type definitions, enums, and utility functions for image retrieval system.
 */

#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <string>
#include "opencv2/opencv.hpp"

/**
 * Feature extraction types
 */
enum class FeatureType {
    BASELINE_5X5,
    BASELINE_7X7,
    BASELINE_9X9,
    HISTOGRAM_RG_8,
    HISTOGRAM_RG_16,
    HISTOGRAM_RGB_8,
    HISTOGRAM_MULTI_RGB_8,
    TEXTURE_COLOR_8,
    TEXTURE_COLOR_GABOR_8,
    TEXTURE_COLOR_LAWS_8,
    TEXTURE_COLOR_FOURIER_8,
    TEXTURE_COLOR_CM_8,
    DEEP_RESNET18,
    CUSTOM_CENTERED_OBJECT,
    CUSTOM_BLUE_SKY,
    FACE_AWARE_RGB_8,
    HISTOGRAM_RG_16_SMOOTH,
    UNKNOWN
};

/**
 * Distance metric types
 */
enum class DistanceMetric {
    SSD,
    L1,
    L_INFINITY,
    HISTOGRAM_INTERSECTION,
    MULTI_HISTOGRAM_INTERSECTION,
    TEXTURE_COLOR,
    TEXTURE_COLOR_GABOR,
    TEXTURE_COLOR_LAWS,
    TEXTURE_COLOR_FOURIER,
    TEXTURE_COLOR_CM,
    COSINE,
    CUSTOM_CENTERED_OBJECT,
    CUSTOM_BLUE_SKY,
    FACE_AWARE,
    UNKNOWN
};

/**
 * Match mode types
 */
enum class MatchMode {
    TOP,
    BOTTOM,
    UNKNOWN
};

/**
 * Parse feature type string to enum
 */
FeatureType parseFeatureType(const char* str);

/**
 * Parse distance metric string to enum
 */
DistanceMetric parseDistanceMetric(const char* str);

/**
 * Parse match mode string to enum
 */
MatchMode parseMatchMode(const char* str);

/**
 * Get feature type as string
 */
const char* featureTypeToString(FeatureType type);

/**
 * Get distance metric as string
 */
const char* distanceMetricToString(DistanceMetric metric);

/**
 * Get expected feature vector size for a given type
 */
size_t getExpectedFeatureSize(FeatureType type);

/**
 * Extract filename from full path
 */
std::string getFilename(const std::string &path);

/**
 * Dispatcher: Extract features based on type
 */
int extractFeatures(cv::Mat &src, FeatureType type, std::vector<float> &features);

/**
 * Dispatcher: Compute distance based on metric
 */
float computeDistance(const std::vector<float> &f1, const std::vector<float> &f2, DistanceMetric metric);

/**
 * Print help message for buildVectorDB
 */
void printBuildDBHelp(const char* programName);

/**
 * Print help message for match
 */
void printMatchHelp(const char* programName);

#endif