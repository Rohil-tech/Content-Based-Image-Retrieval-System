/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Header file for feature extraction functions.
 */

#ifndef FEATURES_H
#define FEATURES_H

#include <vector>
#include "opencv2/opencv.hpp"

/**
 * Extract 5x5 center square as a feature vector.
 * The feature vector will contain 5*5*3 = 75 values (BGR channels).
 * 
 * @param src Source image (must be at least 5x5)
 * @param feature Output feature vector (will be resized to 75 elements)
 * @return 0 on success, -1 on error
 */
int baseline5x5(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 7x7 center square as a feature vector.
 * The feature vector will contain 7*7*3 = 147 values (BGR channels).
 * 
 * @param src Source image (must be at least 7x7)
 * @param feature Output feature vector (will be resized to 147 elements)
 * @return 0 on success, -1 on error
 */ 
int baseline7x7(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 9x9 center square as a feature vector.
 * The feature vector will contain 9*9*3 = 243 values (BGR channels).
 * 
 * @param src Source image (must be at least 9x9)
 * @param feature Output feature vector (will be resized to 243 elements)
 * @return 0 on success, -1 on error
 */
int baseline9x9(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 2D RG chromaticity histogram with 8 bins per channel.
 * Total: 8×8 = 64 values.
 * Chromaticity: r = R/(R+G+B), g = G/(R+G+B)
 * Histogram is normalized (values sum to 1.0).
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 64 elements)
 * @return 0 on success, -1 on error
 */
int histogramRG_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 2D RG chromaticity histogram with 16 bins per channel.
 * Total: 16×16 = 256 values.
 * Chromaticity: r = R/(R+G+B), g = G/(R+G+B)
 * Histogram is normalized (values sum to 1.0).
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 256 elements)
 * @return 0 on success, -1 on error
 */
int histogramRG_16(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 3D RGB color histogram with 8 bins per channel.
 * Total: 8×8×8 = 512 values.
 * Histogram is normalized (values sum to 1.0).
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 512 elements)
 * @return 0 on success, -1 on error
 */
int histogramRGB_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract multi-region RGB histograms (top and bottom halves).
 * Uses 8 bins per channel for each region.
 * Total: 2 regions × 8³ = 1024 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 1024 elements)
 * @return 0 on success, -1 on error
 */
int histogramMultiRGB_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract combined texture and color features.
 * Uses RGB color histogram (8 bins) + Sobel magnitude histogram (8 bins).
 * Total: 512 (RGB) + 8 (texture) = 520 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 520 elements)
 * @return 0 on success, -1 on error
 */
int textureColor_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract combined color and Gabor texture features.
 * Uses RGB color histogram (8 bins) + Gabor filter response histograms.
 * Uses 5 orientations (0°, 36°, 72°, 108°, 144°), 8 bins each.
 * Total: 512 (RGB) + 40 (Gabor) = 552 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 552 elements)
 * @return 0 on success, -1 on error
 */
int textureColorGabor_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract combined color and Laws texture features.
 * Uses RGB color histogram (8 bins) + Laws filter response histograms.
 * Uses 8 common Laws masks (L5E5, E5L5, L5S5, S5L5, E5E5, S5S5, E5S5, S5E5).
 * Total: 512 (RGB) + 64 (Laws) = 576 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 576 elements)
 * @return 0 on success, -1 on error
 */
int textureColorLaws_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract combined color and Fourier texture features.
 * Uses RGB color histogram (8 bins) + Fourier power spectrum (16x16).
 * Total: 512 (RGB) + 256 (Fourier) = 768 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 768 elements)
 * @return 0 on success, -1 on error
 */
int textureColorFourier_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract combined color and CM texture features.
 * Uses RGB color histogram (8 bins) + CM features.
 * Computes 4 directions (0°, 45°, 90°, 135°), 5 features each.
 * Total: 512 (RGB) + 20 (CM) = 532 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 532 elements)
 * @return 0 on success, -1 on error
 */
int textureColorCM_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract centered salient object features (color-agnostic).
 * Detects any prominent object in center regardless of color.
 * Total: 512 (center RGB) + 3 (center-periphery diff) + 1 (concentration) +
 *        8 (saturation) + 6 (background stats) = 530 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 669 elements)
 * @return 0 on success, -1 on error
 */
int customCenteredObjectFeatures(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract blue sky outdoor scene features.
 * Focuses on top region color and texture patterns.
 * Total: 512 (top RGB) + 1 (blueness) + 8 (vertical edges) + 
 *        1 (brightness gradient) + 1 (smoothness) + 8 (blue dominance) + 
 *        16 (cloud texture) = 547 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 547 elements)
 * @return 0 on success, -1 on error
 */
int customBlueSkyFeatures(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract face-aware RGB histograms.
 * If face detected: face histogram (512) + background histogram (512).
 * If no face: whole image histogram (512) + zeros (512).
 * Total: 1024 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 1024 elements)
 * @return 0 on success, -1 on error
 */
int faceAwareRGB_8(cv::Mat &src, std::vector<float> &feature);

/**
 * Extract 2D RG chromaticity histogram with 16 bins and Gaussian smoothing.
 * After building the histogram, applies a Gaussian blur to smooth bin transitions.
 * Total: 16×16 = 256 values.
 * 
 * @param src Source image
 * @param feature Output feature vector (will be resized to 256 elements)
 * @return 0 on success, -1 on error
 */
int histogramRG_16_smooth(cv::Mat &src, std::vector<float> &feature);

#endif