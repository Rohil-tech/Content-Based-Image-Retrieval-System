/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Header file for distance metric functions.
 */

#ifndef DISTANCE_H
#define DISTANCE_H

#include <vector>

/**
 * Compute Sum of Squared Differences (SSD) between two feature vectors.
 * Custom implementation (not using OpenCV).
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector (must be same size as feature1)
 * @return SSD distance value (0 means identical)
 */
float distanceSSD(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute L1 (Manhattan) distance between two feature vectors.
 * L1 = sum of absolute differences.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector (must be same size as feature1)
 * @return L1 distance value (0 means identical)
 */
float distanceL1(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute L-infinity (Chebyshev) distance between two feature vectors.
 * L-inf = maximum absolute difference.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector (must be same size as feature1)
 * @return L-infinity distance value (0 means identical)
 */
float distanceLInf(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute histogram intersection distance between two feature vectors.
 * Both histograms should be normalized.
 * Returns distance = 1 - intersection (so smaller = more similar).
 * 
 * @param feature1 First feature vector (normalized histogram)
 * @param feature2 Second feature vector (normalized histogram)
 * @return Intersection distance value (0 means identical, 1 means no overlap)
 */
float distanceHistogramIntersection(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute multi-histogram intersection distance.
 * Expects feature vectors containing multiple concatenated histograms.
 * Each histogram is compared separately and distances are averaged.
 * 
 * @param feature1 First feature vector (multiple histograms concatenated)
 * @param feature2 Second feature vector
 * @param numHistograms Number of histograms in the feature vector
 * @return Average intersection distance
 */
float distanceMultiHistogramIntersection(const std::vector<float> &feature1, const std::vector<float> &feature2, int numHistograms);

/**
 * Compute texture-color distance.
 * Expects feature vector with RGB histogram (512) + texture histogram (8).
 * Uses histogram intersection for both, with equal weighting.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted average distance
 */
float distanceTextureColor(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute texture-color distance for Gabor features.
 * Expects feature vector with RGB histogram (512) + Gabor histograms (40).
 * Uses histogram intersection for both, with equal weighting.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted average distance
 */
float distanceTextureColorGabor(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute texture-color distance for Laws features.
 * Expects feature vector with RGB histogram (512) + Laws histograms (64).
 * Uses histogram intersection for both, with equal weighting.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted average distance
 */
float distanceTextureColorLaws(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute texture-color distance for Fourier features.
 * Expects feature vector with RGB histogram (512) + Fourier spectrum (256).
 * Uses histogram intersection for color and L2 for Fourier, with equal weighting.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted average distance
 */
float distanceTextureColorFourier(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute texture-color distance for CM features.
 * Expects feature vector with RGB histogram (512) + CM features (20).
 * Uses histogram intersection for color and L2 for CM, with equal weighting.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted average distance
 */
float distanceTextureColorCM(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute cosine distance between two feature vectors.
 * Cosine distance = 1 - cos(theta) where theta is angle between vectors.
 * Works well for high-dimensional embeddings.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Cosine distance (0 = identical direction, 2 = opposite direction)
 */
float distanceCosine(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute custom centered-object distance.
 * Weighted combination of multiple feature components.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted distance
 */
float distanceCustomCenteredObject(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute blue sky scene distance.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted distance
 */
float distanceCustomBlueSky(const std::vector<float> &feature1, const std::vector<float> &feature2);

/**
 * Compute face-aware distance.
 * Expects feature vector with face histogram (512) + background histogram (512).
 * Uses histogram intersection for both, with 60% weight on face, 40% on background.
 * 
 * @param feature1 First feature vector
 * @param feature2 Second feature vector
 * @return Weighted distance
 */
float distanceFaceAware(const std::vector<float> &feature1, const std::vector<float> &feature2);

#endif