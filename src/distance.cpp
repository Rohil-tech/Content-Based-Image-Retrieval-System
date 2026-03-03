/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Implementation of distance metric functions.
 */

#include <cstdio>
#include <vector>
#include <cmath>
#include "distance.h"

/**
 * Compute Sum of Squared Differences (SSD) between two feature vectors.
 */
float distanceSSD(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        printf("Feature1 size: %zu, Feature2 size: %zu\n", feature1.size(), feature2.size());
        return -1.0f;
    }
    
    float ssd = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++) {
        float diff = feature1[i] - feature2[i];
        ssd += diff * diff;
    }
    
    return ssd;
}

/**
 * Compute L1 (Manhattan) distance between two feature vectors.
 */
float distanceL1(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        printf("Feature1 size: %zu, Feature2 size: %zu\n", feature1.size(), feature2.size());
        return -1.0f;
    }
    
    float sum = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++) {
        sum += std::abs(feature1[i] - feature2[i]);
    }
    
    return sum;
}

/**
 * Compute L-infinity (Chebyshev) distance between two feature vectors.
 */
float distanceLInf(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        printf("Feature1 size: %zu, Feature2 size: %zu\n", feature1.size(), feature2.size());
        return -1.0f;
    }
    
    float maxDiff = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++) {
        float diff = std::abs(feature1[i] - feature2[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    
    return maxDiff;
}

/**
 * Compute histogram intersection distance between two feature vectors.
 */
float distanceHistogramIntersection(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        printf("Feature1 size: %zu, Feature2 size: %zu\n", feature1.size(), feature2.size());
        return -1.0f;
    }
    
    float intersection = 0.0f;
    
    for (size_t i = 0; i < feature1.size(); i++) {
        intersection += std::min(feature1[i], feature2[i]);
    }
    
    // Convert to distance: 0 = identical, 1 = no overlap
    float distance = 1.0f - intersection;
    
    return distance;
}

/**
 * Compute multi-histogram intersection distance with equal weighting.
 */
float distanceMultiHistogramIntersection(const std::vector<float> &feature1, const std::vector<float> &feature2, int numHistograms) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() % numHistograms != 0) {
        printf("ERROR: Feature size not divisible by number of histograms\n");
        return -1.0f;
    }
    
    int histSize = feature1.size() / numHistograms;
    float totalDistance = 0.0f;
    
    // Compare each histogram pair
    for (int h = 0; h < numHistograms; h++) {
        float intersection = 0.0f;
        int offset = h * histSize;
        
        for (int i = 0; i < histSize; i++) {
            intersection += std::min(feature1[offset + i], feature2[offset + i]);
        }
        
        // Convert to distance
        float distance = 1.0f - intersection;
        totalDistance += distance;
    }
    
    // Equal weighting: average the distances
    return totalDistance / numHistograms;
}

/**
 * Compute texture-color distance with equal weighting.
 */
float distanceTextureColor(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 520) {
        printf("ERROR: Expected 520 values for texture-color features\n");
        return -1.0f;
    }
    
    // Compare color histogram (first 512 values)
    float colorIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        colorIntersection += std::min(feature1[i], feature2[i]);
    }
    float colorDistance = 1.0f - colorIntersection;
    
    // Compare texture histogram (last 8 values)
    float textureIntersection = 0.0f;
    for (int i = 512; i < 520; i++) {
        textureIntersection += std::min(feature1[i], feature2[i]);
    }
    float textureDistance = 1.0f - textureIntersection;
    
    // Equal weighting: average the two distances
    return (colorDistance + textureDistance) / 2.0f;
}

/**
 * Compute texture-color distance with Gabor features.
 */
float distanceTextureColorGabor(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        printf("Feature1 size: %zu, Feature2 size: %zu\n", feature1.size(), feature2.size());
        return -1.0f;
    }
    
    if (feature1.size() != 552) {
        printf("ERROR: Expected 552 values for texture-color-gabor features\n");
        printf("Got: %zu values\n", feature1.size());
        return -1.0f;
    }
    
    // Compare color histogram (first 512 values)
    float colorIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        colorIntersection += std::min(feature1[i], feature2[i]);
    }
    float colorDistance = 1.0f - colorIntersection;
    
    // Compare Gabor texture histogram (last 40 values)
    float textureIntersection = 0.0f;
    for (int i = 512; i < 552; i++) {
        textureIntersection += std::min(feature1[i], feature2[i]);
    }
    float textureDistance = 1.0f - textureIntersection;
    
    // Equal weighting
    return (colorDistance + textureDistance) / 2.0f;
}

/**
 * Compute texture-color distance with Laws features.
 */
float distanceTextureColorLaws(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 576) {
        printf("ERROR: Expected 576 values for texture-color-laws features\n");
        printf("Got: %zu values\n", feature1.size());
        return -1.0f;
    }
    
    // Compare color histogram (first 512 values)
    float colorIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        colorIntersection += std::min(feature1[i], feature2[i]);
    }
    float colorDistance = 1.0f - colorIntersection;
    
    // Compare Laws texture histogram (last 64 values)
    float textureIntersection = 0.0f;
    for (int i = 512; i < 576; i++) {
        textureIntersection += std::min(feature1[i], feature2[i]);
    }
    float textureDistance = 1.0f - textureIntersection;
    
    // Equal weighting
    float finalDistance = (colorDistance + textureDistance) / 2.0f;
    return finalDistance;
}

/**
 * Compute texture-color distance with Fourier features.
 */
float distanceTextureColorFourier(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 768) {
        printf("ERROR: Expected 768 values for texture-color-fourier features\n");
        return -1.0f;
    }
    
    // Compare color histogram (first 512 values) using intersection
    float colorIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        colorIntersection += std::min(feature1[i], feature2[i]);
    }
    float colorDistance = 1.0f - colorIntersection;
    
    // Compare Fourier spectrum (last 256 values) using normalized L2
    float fourierSSD = 0.0f;
    for (int i = 512; i < 768; i++) {
        float diff = feature1[i] - feature2[i];
        fourierSSD += diff * diff;
    }
    float fourierDistance = std::sqrt(fourierSSD) / std::sqrt(256.0f);  // Normalize by max possible
    
    // Equal weighting
    return (colorDistance + fourierDistance) / 2.0f;
}

/**
 * Compute texture-color distance with CM features.
 */
float distanceTextureColorCM(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 532) {
        printf("ERROR: Expected 532 values for texture-color-cm features\n");
        return -1.0f;
    }
    
    // Compare color histogram (first 512 values) using intersection
    float colorIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        colorIntersection += std::min(feature1[i], feature2[i]);
    }
    float colorDistance = 1.0f - colorIntersection;
    
    // Compare CM features (last 20 values) using L2
    float cmSSD = 0.0f;
    for (int i = 512; i < 532; i++) {
        float diff = feature1[i] - feature2[i];
        cmSSD += diff * diff;
    }
    float cmDistance = std::sqrt(cmSSD) / std::sqrt(20.0f);  // Normalize
    
    // Equal weighting
    return (colorDistance + cmDistance) / 2.0f;
}

/**
 * Compute cosine distance between two feature vectors.
 */
float distanceCosine(const std::vector<float> &feature1, 
                     const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    // Compute magnitudes (L2 norms)
    float mag1 = 0.0f;
    float mag2 = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        mag1 += feature1[i] * feature1[i];
        mag2 += feature2[i] * feature2[i];
    }
    mag1 = std::sqrt(mag1);
    mag2 = std::sqrt(mag2);
    
    // Avoid division by zero
    if (mag1 < 1e-10f || mag2 < 1e-10f) {
        return 1.0f;  // Return maximum distance if either vector is zero
    }
    
    // Compute dot product of normalized vectors
    float dotProduct = 0.0f;
    for (size_t i = 0; i < feature1.size(); i++) {
        dotProduct += (feature1[i] / mag1) * (feature2[i] / mag2);
    }
    
    // Clamp to [-1, 1] to handle floating point errors
    if (dotProduct > 1.0f) dotProduct = 1.0f;
    if (dotProduct < -1.0f) dotProduct = -1.0f;
    
    // Cosine distance = 1 - cos(theta)
    float cosineDistance = 1.0f - dotProduct;
    
    return cosineDistance;
}

/**
 * Compute centered object distance (generic, color-agnostic).
 */
float distanceCustomCenteredObject(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 530) {
        printf("ERROR: Expected 530 values\n");
        return -1.0f;
    }
    
    // Center histogram (512) - 40% weight
    float histIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        histIntersection += std::min(feature1[i], feature2[i]);
    }
    float histDistance = 1.0f - histIntersection;
    
    // Center-periphery difference (3) - 20% weight
    float colorDiff = 0.0f;
    for (int i = 512; i < 515; i++) {
        colorDiff += std::abs(feature1[i] - feature2[i]);
    }
    colorDiff /= 3.0f;
    
    // Saturation concentration (1) - 15% weight
    float satDiff = std::abs(feature1[515] - feature2[515]);
    
    // Saturation histogram (8) - 15% weight
    float satIntersection = 0.0f;
    for (int i = 516; i < 524; i++) {
        satIntersection += std::min(feature1[i], feature2[i]);
    }
    float satDistance = 1.0f - satIntersection;
    
    // Background stats (6) - 10% weight
    float bgDiff = 0.0f;
    for (int i = 524; i < 530; i++) {
        bgDiff += std::abs(feature1[i] - feature2[i]);
    }
    bgDiff /= 6.0f;
    
    // Weighted combination
    return 0.40f * histDistance +
           0.20f * colorDiff +
           0.15f * satDiff +
           0.15f * satDistance +
           0.10f * bgDiff;
}

/**
 * Compute blue sky scene distance.
 */
float distanceCustomBlueSky(const std::vector<float> &feature1, const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 547) {
        printf("ERROR: Expected 547 values for blue sky features\n");
        printf("Got: %zu values\n", feature1.size());
        return -1.0f;
    }
    
    // Component 1: Top RGB histogram (512) - 40% weight
    float histIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        histIntersection += std::min(feature1[i], feature2[i]);
    }
    float histDistance = 1.0f - histIntersection;
    
    // Component 2: Blueness score (1) - 25% weight
    float bluenessDiff = std::abs(feature1[512] - feature2[512]);
    
    // Component 3: Vertical edges (8) - 10% weight
    float edgeIntersection = 0.0f;
    for (int i = 513; i < 521; i++) {
        edgeIntersection += std::min(feature1[i], feature2[i]);
    }
    float edgeDistance = 1.0f - edgeIntersection;
    
    // Component 4: Brightness gradient (1) - 10% weight
    float gradientDiff = std::abs(feature1[521] - feature2[521]);
    
    // Component 5: Smoothness (1) - 5% weight
    float smoothDiff = std::abs(feature1[522] - feature2[522]);
    
    // Component 6: Blue dominance histogram (8) - 5% weight
    float blueDomIntersection = 0.0f;
    for (int i = 523; i < 531; i++) {
        blueDomIntersection += std::min(feature1[i], feature2[i]);
    }
    float blueDomDistance = 1.0f - blueDomIntersection;
    
    // Component 7: Cloud texture (16) - 5% weight
    float cloudIntersection = 0.0f;
    for (int i = 531; i < 547; i++) {
        cloudIntersection += std::min(feature1[i], feature2[i]);
    }
    float cloudDistance = 1.0f - cloudIntersection;
    
    // Weighted combination
    float totalDistance = 
        0.40f * histDistance +
        0.25f * bluenessDiff +
        0.10f * edgeDistance +
        0.10f * gradientDiff +
        0.05f * smoothDiff +
        0.05f * blueDomDistance +
        0.05f * cloudDistance;
    
    return totalDistance;
}

/**
 * Compute face-aware distance with weighted histograms.
 */
float distanceFaceAware(const std::vector<float> &feature1, 
                        const std::vector<float> &feature2) {
    if (feature1.size() != feature2.size()) {
        printf("ERROR: Feature vectors must be the same size\n");
        return -1.0f;
    }
    
    if (feature1.size() != 1024) {
        printf("ERROR: Expected 1024 values for face-aware features\n");
        return -1.0f;
    }
    
    // Compare face histogram (first 512 values)
    float faceIntersection = 0.0f;
    for (int i = 0; i < 512; i++) {
        faceIntersection += std::min(feature1[i], feature2[i]);
    }
    float faceDistance = 1.0f - faceIntersection;
    
    // Compare background histogram (last 512 values)
    float bgIntersection = 0.0f;
    for (int i = 512; i < 1024; i++) {
        bgIntersection += std::min(feature1[i], feature2[i]);
    }
    float bgDistance = 1.0f - bgIntersection;
    
    // Weighted combination: 95% face, 5% background
    return 0.9f * faceDistance + 0.1f * bgDistance;
}