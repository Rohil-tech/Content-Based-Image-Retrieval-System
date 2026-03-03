/**
 * Rohil Kulshreshtha
 * January 31, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Implementation of type utilities and dispatcher functions.
 */

#include <cstring>
#include <cstdio>
#include <algorithm>
#include "types.h"
#include "features.h"
#include "distance.h"

/**
 * Parse feature type string to enum
 */
FeatureType parseFeatureType(const char* str) {
    if (strcmp(str, "baseline_5x5") == 0) return FeatureType::BASELINE_5X5;
    if (strcmp(str, "baseline_7x7") == 0) return FeatureType::BASELINE_7X7;
    if (strcmp(str, "baseline_9x9") == 0) return FeatureType::BASELINE_9X9;
    if (strcmp(str, "histogram_rg_8") == 0) return FeatureType::HISTOGRAM_RG_8;
    if (strcmp(str, "histogram_rg_16") == 0) return FeatureType::HISTOGRAM_RG_16;
    if (strcmp(str, "histogram_rgb_8") == 0) return FeatureType::HISTOGRAM_RGB_8;
    if (strcmp(str, "histogram_multi_rgb_8") == 0) return FeatureType::HISTOGRAM_MULTI_RGB_8;
    if (strcmp(str, "texture_color_8") == 0) return FeatureType::TEXTURE_COLOR_8;
    if (strcmp(str, "texture_color_gabor_8") == 0) return FeatureType::TEXTURE_COLOR_GABOR_8;
    if (strcmp(str, "texture_color_laws_8") == 0) return FeatureType::TEXTURE_COLOR_LAWS_8;
    if (strcmp(str, "texture_color_fourier_8") == 0) return FeatureType::TEXTURE_COLOR_FOURIER_8;
    if (strcmp(str, "texture_color_cm_8") == 0) return FeatureType::TEXTURE_COLOR_CM_8;
    if (strcmp(str, "deep_resnet18") == 0) return FeatureType::DEEP_RESNET18;
    if (strcmp(str, "custom_centered_object") == 0) return FeatureType::CUSTOM_CENTERED_OBJECT;
    if (strcmp(str, "custom_blue_sky") == 0) return FeatureType::CUSTOM_BLUE_SKY;
    if (strcmp(str, "face_aware_rgb_8") == 0) return FeatureType::FACE_AWARE_RGB_8;
    if (strcmp(str, "histogram_rg_16_smooth") == 0) return FeatureType::HISTOGRAM_RG_16_SMOOTH;
    return FeatureType::UNKNOWN;
}

/**
 * Parse distance metric string to enum
 */
DistanceMetric parseDistanceMetric(const char* str) {
    if (strcmp(str, "ssd") == 0) return DistanceMetric::SSD;
    if (strcmp(str, "l1") == 0) return DistanceMetric::L1;
    if (strcmp(str, "linf") == 0) return DistanceMetric::L_INFINITY;
    if (strcmp(str, "intersection") == 0) return DistanceMetric::HISTOGRAM_INTERSECTION;
    if (strcmp(str, "multi_intersection") == 0) return DistanceMetric::MULTI_HISTOGRAM_INTERSECTION;
    if (strcmp(str, "texture_color") == 0) return DistanceMetric::TEXTURE_COLOR;
    if (strcmp(str, "texture_color_gabor") == 0) return DistanceMetric::TEXTURE_COLOR_GABOR;
    if (strcmp(str, "texture_color_laws") == 0) return DistanceMetric::TEXTURE_COLOR_LAWS;
    if (strcmp(str, "texture_color_fourier") == 0) return DistanceMetric::TEXTURE_COLOR_FOURIER;
    if (strcmp(str, "texture_color_cm") == 0) return DistanceMetric::TEXTURE_COLOR_CM;
    if (strcmp(str, "cosine") == 0) return DistanceMetric::COSINE;
    if (strcmp(str, "custom_centered_object") == 0) return DistanceMetric::CUSTOM_CENTERED_OBJECT;
    if (strcmp(str, "custom_blue_sky") == 0) return DistanceMetric::CUSTOM_BLUE_SKY;
    if (strcmp(str, "face_aware") == 0) return DistanceMetric::FACE_AWARE;
    return DistanceMetric::UNKNOWN;
}

/**
 * Parse match mode string to enum
 */
MatchMode parseMatchMode(const char* str) {
    if (strcmp(str, "top") == 0) return MatchMode::TOP;
    if (strcmp(str, "bottom") == 0) return MatchMode::BOTTOM;
    return MatchMode::UNKNOWN;
}

/**
 * Get feature type as string
 */
const char* featureTypeToString(FeatureType type) {
    switch(type) {
        case FeatureType::BASELINE_5X5: return "baseline_5x5";
        case FeatureType::BASELINE_7X7: return "baseline_7x7";
        case FeatureType::BASELINE_9X9: return "baseline_9x9";
        case FeatureType::HISTOGRAM_RG_8: return "histogram_rg_8";
        case FeatureType::HISTOGRAM_RG_16: return "histogram_rg_16";
        case FeatureType::HISTOGRAM_RGB_8: return "histogram_rgb_8";
        case FeatureType::HISTOGRAM_MULTI_RGB_8: return "histogram_multi_rgb_8";
        case FeatureType::TEXTURE_COLOR_8: return "texture_color_8";
        case FeatureType::TEXTURE_COLOR_GABOR_8: return "texture_color_gabor_8";
        case FeatureType::TEXTURE_COLOR_LAWS_8: return "texture_color_laws_8";
        case FeatureType::TEXTURE_COLOR_FOURIER_8: return "texture_color_fourier_8";
        case FeatureType::TEXTURE_COLOR_CM_8: return "texture_color_cm_8";
        case FeatureType::DEEP_RESNET18: return "deep_resnet18";
        case FeatureType::CUSTOM_CENTERED_OBJECT: return "custom_centered_object";
        case FeatureType::CUSTOM_BLUE_SKY: return "custom_blue_sky";
        case FeatureType::FACE_AWARE_RGB_8: return "face_aware_rgb_8";
        case FeatureType::HISTOGRAM_RG_16_SMOOTH: return "histogram_rg_16_smooth";
        default: return "unknown";
    }
}

/**
 * Get distance metric as string
 */
const char* distanceMetricToString(DistanceMetric metric) {
    switch(metric) {
        case DistanceMetric::SSD: return "ssd";
        case DistanceMetric::L1: return "l1";
        case DistanceMetric::L_INFINITY: return "linf";
        case DistanceMetric::HISTOGRAM_INTERSECTION: return "intersection";
        case DistanceMetric::MULTI_HISTOGRAM_INTERSECTION: return "multi_intersection";
        case DistanceMetric::TEXTURE_COLOR: return "texture_color";
        case DistanceMetric::TEXTURE_COLOR_GABOR: return "texture_color_gabor";
        case DistanceMetric::TEXTURE_COLOR_LAWS: return "texture_color_laws";
        case DistanceMetric::TEXTURE_COLOR_FOURIER: return "texture_color_fourier";
        case DistanceMetric::TEXTURE_COLOR_CM: return "texture_color_cm";
        case DistanceMetric::COSINE: return "cosine";
        case DistanceMetric::CUSTOM_CENTERED_OBJECT: return "custom_centered_object";
        case DistanceMetric::CUSTOM_BLUE_SKY: return "custom_blue_sky";
        case DistanceMetric::FACE_AWARE: return "face_aware";
        default: return "unknown";
    }
}

/**
 * Get expected feature vector size for a given type
 */
size_t getExpectedFeatureSize(FeatureType type) {
    switch(type) {
        case FeatureType::BASELINE_5X5: return 5 * 5 * 3;  // 75
        case FeatureType::BASELINE_7X7: return 7 * 7 * 3;  // 147
        case FeatureType::BASELINE_9X9: return 9 * 9 * 3;  // 243
        case FeatureType::HISTOGRAM_RG_8: return 8 * 8;  // 64
        case FeatureType::HISTOGRAM_RG_16: return 16 * 16;  // 256
        case FeatureType::HISTOGRAM_RGB_8: return 8 * 8 * 8;  // 512
        case FeatureType::HISTOGRAM_MULTI_RGB_8: return 2 * 8 * 8 * 8; // 2 Histograms + 512 features each
        case FeatureType::TEXTURE_COLOR_8: return 8 * 8 * 8 + 8;  // 512 (RGB) + 8 (magnitude)
        case FeatureType::TEXTURE_COLOR_GABOR_8: return 8 * 8 * 8 + 40;  // 512 (RGB) + 40 (5 orientations × 8 bins)
        case FeatureType::TEXTURE_COLOR_LAWS_8: return 8 * 8 * 8 + 64;  // 512 (RGB) + 64 (8 Laws masks × 8 bins)
        case FeatureType::TEXTURE_COLOR_FOURIER_8: return 8 * 8 * 8 + 16 * 16;  // 512 (RGB) + 256 (16x16 power spectrum)
        case FeatureType::TEXTURE_COLOR_CM_8: return 8 * 8 * 8 + 20;  // 512 (RGB) + 20 (4 directions × 5 features)
        case FeatureType::DEEP_RESNET18: return 512;  // ResNet18 final layer output
        case FeatureType::CUSTOM_CENTERED_OBJECT: return 512 + 3 + 1 + 8 + 6;  // 530 values
        case FeatureType::CUSTOM_BLUE_SKY: return 512 + 1 + 8 + 1 + 1 + 8 + 16;  // 547 values
        case FeatureType::FACE_AWARE_RGB_8: return 2 * 8 * 8 * 8;  // 1024 (face + background histograms)
        case FeatureType::HISTOGRAM_RG_16_SMOOTH: return 16 * 16;  // Same as regular, just smoothed
        default: return 0;
    }
}

/**
 * Extract filename from full path
 */
std::string getFilename(const std::string &path) {
    // Find last occurrence of '/' or '\'
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos) {
        return path.substr(pos + 1);
    }
    return path;
}

/**
 * Dispatcher: Extract features based on type
 */
int extractFeatures(cv::Mat &src, FeatureType type, std::vector<float> &features) {
    switch(type) {
        case FeatureType::BASELINE_5X5:
            return baseline5x5(src, features);
        
        case FeatureType::BASELINE_7X7:
            return baseline7x7(src, features);
        
        case FeatureType::BASELINE_9X9:
            return baseline9x9(src, features);
        
        case FeatureType::HISTOGRAM_RG_8:
            return histogramRG_8(src, features);
        
        case FeatureType::HISTOGRAM_RG_16:
            return histogramRG_16(src, features);
        
        case FeatureType::HISTOGRAM_RGB_8:
            return histogramRGB_8(src, features);

        case FeatureType::HISTOGRAM_MULTI_RGB_8:
            return histogramMultiRGB_8(src, features);
        
        case FeatureType::TEXTURE_COLOR_8:
            return textureColor_8(src, features);
        
        case FeatureType::TEXTURE_COLOR_GABOR_8:
            return textureColorGabor_8(src, features);
        
        case FeatureType::TEXTURE_COLOR_LAWS_8:
            return textureColorLaws_8(src, features);
        
        case FeatureType::TEXTURE_COLOR_FOURIER_8:
            return textureColorFourier_8(src, features);
        
        case FeatureType::TEXTURE_COLOR_CM_8:
            return textureColorCM_8(src, features);
        
        case FeatureType::CUSTOM_CENTERED_OBJECT:
            return customCenteredObjectFeatures(src, features);
        
        case FeatureType::CUSTOM_BLUE_SKY:
            return customBlueSkyFeatures(src, features);
        
        case FeatureType::FACE_AWARE_RGB_8:
            return faceAwareRGB_8(src, features);
        
        case FeatureType::HISTOGRAM_RG_16_SMOOTH:
            return histogramRG_16_smooth(src, features);
        
        default:
            printf("ERROR: Unknown feature type\n");
            return -1;
    }
}

/**
 * Dispatcher: Compute distance based on metric
 */
float computeDistance(const std::vector<float> &f1, const std::vector<float> &f2, DistanceMetric metric) {
    switch(metric) {
        case DistanceMetric::SSD:
            return distanceSSD(f1, f2);
        
        case DistanceMetric::L1:
            return distanceL1(f1, f2);
        
        case DistanceMetric::L_INFINITY:
            return distanceLInf(f1, f2);
        
        case DistanceMetric::HISTOGRAM_INTERSECTION:
            return distanceHistogramIntersection(f1, f2);
        
        case DistanceMetric::MULTI_HISTOGRAM_INTERSECTION:
            return distanceMultiHistogramIntersection(f1, f2, 2);
        
        case DistanceMetric::TEXTURE_COLOR:
            return distanceTextureColor(f1, f2);
        
        case DistanceMetric::TEXTURE_COLOR_GABOR:
            return distanceTextureColorGabor(f1, f2);
        
        case DistanceMetric::TEXTURE_COLOR_LAWS:
            return distanceTextureColorLaws(f1, f2);
        
        case DistanceMetric::TEXTURE_COLOR_FOURIER:
            return distanceTextureColorFourier(f1, f2);
        
        case DistanceMetric::TEXTURE_COLOR_CM:
            return distanceTextureColorCM(f1, f2);
        
        case DistanceMetric::COSINE:
            return distanceCosine(f1, f2);
        
        case DistanceMetric::CUSTOM_CENTERED_OBJECT:
            return distanceCustomCenteredObject(f1, f2);
        
        case DistanceMetric::CUSTOM_BLUE_SKY:
            return distanceCustomBlueSky(f1, f2);
        
        case DistanceMetric::FACE_AWARE:
            return distanceFaceAware(f1, f2);
        
        default:
            printf("ERROR: Unknown distance metric\n");
            return -1.0f;
    }
}

/**
 * Print help message for buildVectorDB
 */
void printBuildDBHelp(const char* programName) {
    printf("\n=== buildVectorDB - Feature Database Builder ===\n\n");
    printf("Usage: %s <directory_path> <feature_type> <output_csv>\n\n", programName);
    
    printf("Feature Types:\n");
    printf("  baseline_5x5 - 5x5 center square (75 values)\n");
    printf("  baseline_7x7 - 7x7 center square (147 values)\n");
    printf("  baseline_9x9 - 9x9 center square (243 values)\n");
    printf("  histogram_rg_8 - RG chromaticity, 8 bins (64 values)\n");
    printf("  histogram_rg_16 - RG chromaticity, 16 bins (256 values)\n");
    printf("  histogram_rgb_8 - RGB color, 8 bins (512 values)\n");
    printf("  histogram_multi_rgb_8 - Top/bottom RGB histograms, 8 bins (1024 values)\n");
    printf("  texture_color_8 - RGB + texture histograms, 8 bins (520 values)\n");
    printf("  texture_color_gabor_8 - RGB + Gabor texture, 8 bins (552 values)\n");
    printf("  texture_color_laws_8 - RGB + Laws texture, 8 bins (576 values)\n");
    printf("  texture_color_fourier_8 - RGB + Fourier texture, 8 bins (768 values)\n");
    printf("  texture_color_cm_8 - RGB + CM texture, 8 bins (532 values)\n");
    printf("  deep_resnet18 - Pre-trained ResNet18 embeddings (512 values)\n");
    printf("  custom_centered_object - Centered objects-specific detector (530 values)\n");
    printf("  custom_blue_sky - Blue sky outdoor scene detector (547 values)\n");
    printf("  face_aware_rgb_8 - Face + background RGB, 8 bins (1024 values)\n");
    printf("  histogram_rg_16_smooth - RG chromaticity, 16 bins, smoothed (256 values)\n\n");
    
    printf("Examples:\n");
    printf("  %s data/olympus baseline_7x7 data/features/baseline_7x7.csv\n", programName);
    printf("  %s data/olympus histogram_rg_16 data/features/hist_rg_16.csv\n\n", programName);
}

/**
 * Print help message for match
 */
void printMatchHelp(const char* programName) {
    printf("\n=== match - Image Matching Tool ===\n\n");
    printf("Usage: %s <target_image> <feature_csv> <feature_type> <distance_metric> <num_matches> [match_mode]\n\n", programName);
    
    printf("Feature Types:\n");
    printf("  baseline_5x5 - 5x5 center square (75 values)\n");
    printf("  baseline_7x7 - 7x7 center square (147 values)\n");
    printf("  baseline_9x9 - 9x9 center square (243 values)\n");
    printf("  histogram_rg_8 - RG chromaticity, 8 bins (64 values)\n");
    printf("  histogram_rg_16 - RG chromaticity, 16 bins (256 values)\n");
    printf("  histogram_rgb_8 - RGB color, 8 bins (512 values)\n");
    printf("  histogram_multi_rgb_8 - Top/bottom RGB histograms, 8 bins (1024 values)\n");
    printf("  texture_color_8 - RGB + texture histograms, 8 bins (520 values)\n");
    printf("  texture_color_gabor_8 - RGB + Gabor texture, 8 bins (552 values)\n");
    printf("  texture_color_fourier_8 - RGB + Fourier texture, 8 bins (768 values)\n");
    printf("  texture_color_cm_8 - RGB + CM texture, 8 bins (532 values)\n");
    printf("  deep_resnet18 - Pre-trained ResNet18 embeddings (512 values)\n");
    printf("  custom_centered_object - Centered objects-specific detector (530 values)\n");
    printf("  custom_blue_sky - Blue sky outdoor scene detector (547 values)\n");
    printf("  face_aware_rgb_8 - Face + background RGB, 8 bins (1024 values)\n");
    printf("  histogram_rg_16_smooth - RG chromaticity, 16 bins, smoothed (256 values)\n\n");
    
    printf("Distance Metrics:\n");
    printf("  ssd - Sum of Squared Differences\n");
    printf("  l1 - Manhattan distance (L1)\n");
    printf("  linf - Chebyshev distance (L-infinity)\n");
    printf("  intersection - Histogram intersection\n");
    printf("  multi_intersection - Multi-histogram intersection\n");
    printf("  texture_color - Texture + color distance\n\n");
    printf("  texture_color_gabor - Texture (Gabor) + color distance\n");
    printf("  texture_color_laws - Texture (Laws) + color distance\n");
    printf("  texture_color_fourier - Texture (Fourier) + color distance\n");
    printf("  texture_color_cm - Texture (CM) + color distance\n");
    printf("  cosine - Cosine distance (for embeddings)\n");
    printf("  custom_centered_object - Centered Objects-specific distance\n");
    printf("  custom_blue_sky - Blue sky scene distance\n");
    printf("  face_aware - Face + background weighted distance\n\n");
    
    printf("Match Modes (optional, default: top):\n");
    printf("  top - Most similar images (smallest distance)\n");
    printf("  bottom - Least similar images (largest distance)\n\n");
    
    printf("Recommended Feature-Metric Pairings:\n");
    printf("  baseline_* → ssd, l1, linf\n");
    printf("  histogram_* → intersection\n\n");
    
    printf("Examples:\n");
    printf("  %s data/olympus/pic.1016.jpg data/features/baseline_7x7.csv baseline_7x7 ssd 4\n", programName);
    printf("  %s data/olympus/pic.0164.jpg data/features/hist_rg_16.csv histogram_rg_16 intersection 3\n", programName);
    printf("  %s data/olympus/pic.1016.jpg data/features/baseline_7x7.csv baseline_7x7 ssd 5 bottom\n\n", programName);
}