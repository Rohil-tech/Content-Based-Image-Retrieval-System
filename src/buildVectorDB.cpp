/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Program to extract features from all images in a directory and save to CSV.
 * 
 * Usage: buildVectorDB <directory_path> <feature_type> <output_csv>
 * Example: ./bin/buildVectorDB data/olympus histogram_rg_16 data/features/hist_rg_16.csv
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include "dirent.h"
#include "opencv2/opencv.hpp"
#include "types.h"
#include "csv_util.h"

int main(int argc, char *argv[]) {

    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);
    
    char dirname[256];
    char buffer[256];
    char featureTypeStr[256];
    char outputCSV[256];
    DIR *dirp;
    struct dirent *dp;
    
    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printBuildDBHelp(argv[0]);
        return 0;
    }
    
    if (argc < 4) {
        printf("ERROR: Insufficient arguments\n");
        printBuildDBHelp(argv[0]);
        return -1;
    }
    
    strcpy(dirname, argv[1]);
    strcpy(featureTypeStr, argv[2]);
    strcpy(outputCSV, argv[3]);
    
    FeatureType featureType = parseFeatureType(featureTypeStr);
    if (featureType == FeatureType::UNKNOWN) {
        printf("ERROR: Unknown feature type '%s'\n", featureTypeStr);
        printBuildDBHelp(argv[0]);
        return -1;
    }
    
    printf("=== Feature Extraction ===\n");
    printf("Directory: %s\n", dirname);
    printf("Feature type: %s\n", featureTypeToString(featureType));
    printf("Expected feature size: %zu values\n", getExpectedFeatureSize(featureType));
    printf("Output CSV: %s\n\n", outputCSV);
    
    dirp = opendir(dirname);
    if (dirp == NULL) {
        printf("ERROR: Cannot open directory %s\n", dirname);
        return -1;
    }
    
    int imageCount = 0;
    int resetFile = 1;
    
    while ((dp = readdir(dirp)) != NULL) {
        
        // Check if file is an image
        if (strstr(dp->d_name, ".jpg") ||
            strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") ||
            strstr(dp->d_name, ".tif")) {
            
            // Build full path
            strcpy(buffer, dirname);
            strcat(buffer, "/");
            strcat(buffer, dp->d_name);
            
            cv::Mat image = cv::imread(buffer);
            if (image.data == NULL) {
                printf("WARNING: Could not read image %s\n", buffer);
                continue;
            }
            
            // Extract features
            std::vector<float> features;
            int result = extractFeatures(image, featureType, features);
            
            if (result != 0) {
                printf("WARNING: Feature extraction failed for %s\n", buffer);
                continue;
            }
            
            // Validate feature size
            size_t expectedSize = getExpectedFeatureSize(featureType);
            if (features.size() != expectedSize) {
                printf("WARNING: Feature size mismatch for %s (expected %zu, got %zu)\n", 
                       buffer, expectedSize, features.size());
                continue;
            }
            
            append_image_data_csv(outputCSV, buffer, features, resetFile);
            resetFile = 0;
            
            imageCount++;
            if (imageCount % 50 == 0) {
                printf("Processed %d images...\n", imageCount);
            }
        }
    }
    
    closedir(dirp);
    
    printf("\n=== Feature Extraction Complete ===\n");
    printf("Total images processed: %d\n", imageCount);
    printf("Features saved to: %s\n\n\n", outputCSV);
    
    return 0;
}