/**
 * Rohil Kulshreshtha
 * January 30, 2026
 * CS 5330 - PR-CV - Assignment 2
 * 
 * Program to find the top N matches for a target image using pre-computed features.
 * 
 * Usage: match <target_image> <feature_csv> <feature_type> <distance_metric> <num_matches> [match_mode]
 * Example: ./bin/match data/olympus/pic.1016.jpg data/features/baseline_7x7.csv baseline_7x7 ssd 4
 * Example: ./bin/match data/olympus/pic.1016.jpg data/features/baseline_7x7.csv baseline_7x7 ssd 4 bottom
 */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <string>
#include "opencv2/opencv.hpp"
#include "types.h"
#include "csv_util.h"

struct Match {
    char *filename;
    float distance;
};

int main(int argc, char *argv[]) {
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_WARNING);

    if (argc == 2 && (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0)) {
        printMatchHelp(argv[0]);
        return 0;
    }
    
    if (argc < 6) {
        printf("ERROR: Insufficient arguments\n");
        printMatchHelp(argv[0]);
        return -1;
    }
    
    char targetImage[256];
    char featureCSV[256];
    char featureTypeStr[256];
    char distanceMetricStr[256];
    int numMatches;
    
    strcpy(targetImage, argv[1]);
    strcpy(featureCSV, argv[2]);
    strcpy(featureTypeStr, argv[3]);
    strcpy(distanceMetricStr, argv[4]);
    numMatches = atoi(argv[5]);
    
    MatchMode matchMode = MatchMode::TOP;
    if (argc >= 7) {
        matchMode = parseMatchMode(argv[6]);
        if (matchMode == MatchMode::UNKNOWN) {
            printf("ERROR: Unknown match mode '%s'\n", argv[6]);
            printMatchHelp(argv[0]);
            return -1;
        }
    }
    
    FeatureType featureType = parseFeatureType(featureTypeStr);
    if (featureType == FeatureType::UNKNOWN) {
        printf("ERROR: Unknown feature type '%s'\n", featureTypeStr);
        printMatchHelp(argv[0]);
        return -1;
    }
    
    DistanceMetric distanceMetric = parseDistanceMetric(distanceMetricStr);
    if (distanceMetric == DistanceMetric::UNKNOWN) {
        printf("ERROR: Unknown distance metric '%s'\n", distanceMetricStr);
        printMatchHelp(argv[0]);
        return -1;
    }
    
    size_t expectedSize = getExpectedFeatureSize(featureType);
    
    printf("=== Image Matching ===\n");
    printf("Target image: %s\n", targetImage);
    printf("Feature CSV: %s\n", featureCSV);
    printf("Feature type: %s\n", featureTypeToString(featureType));
    printf("Distance metric: %s\n", distanceMetricToString(distanceMetric));
    printf("Number of matches: %d\n", numMatches);
    printf("Match mode: %s\n\n", matchMode == MatchMode::TOP ? "top" : "bottom");
    
    cv::Mat target = cv::imread(targetImage);
    if (target.data == NULL) {
        printf("ERROR: Could not read target image %s\n", targetImage);
        return -1;
    }

    // Load feature database
    std::vector<char *> filenames;
    std::vector<std::vector<float>> database;
    int result = read_image_data_csv(featureCSV, filenames, database, 0);
    
    if (result != 0) {
        printf("ERROR: Could not read feature database\n");
        return -1;
    }

    printf("Database loaded: %zu images\n", filenames.size());

    // Validate database feature sizes
    if (database.size() > 0 && database[0].size() != expectedSize) {
        printf("ERROR: Database feature size mismatch!\n");
        printf("Expected %zu values (for %s), but CSV contains %zu values\n",
               expectedSize, featureTypeToString(featureType), database[0].size());
        printf("Did you use the correct feature type when building the database?\n");
        return -1;
    }

    // Extract or load target features
    std::vector<float> targetFeatures;

    if (featureType == FeatureType::DEEP_RESNET18) {
        // For deep features, load from CSV
        std::string targetFilename = getFilename(targetImage);
        bool found = false;
        
        for (size_t i = 0; i < filenames.size(); i++) {
            std::string dbFilename = getFilename(filenames[i]);
            if (targetFilename == dbFilename) {
                targetFeatures = database[i];
                found = true;
                break;
            }
        }
        
        if (!found) {
            printf("ERROR: Target image not found in deep feature database\n");
            return -1;
        }
    } else {
        // For all other features, compute from image
        result = extractFeatures(target, featureType, targetFeatures);
        
        if (result != 0) {
            printf("ERROR: Feature extraction failed for target image\n");
            return -1;
        }
    }

    // Validate target feature size
    if (targetFeatures.size() != expectedSize) {
        printf("ERROR: Target feature size mismatch (expected %zu, got %zu)\n", 
               expectedSize, targetFeatures.size());
        return -1;
    }

    printf("Target features extracted: %zu values\n\n", targetFeatures.size());
    
    // Calculate distances
    std::vector<Match> matches;
    std::string targetFilename = getFilename(targetImage);
    
    for (size_t i = 0; i < filenames.size(); i++) {
        std::string dbFilename = getFilename(filenames[i]);
        
        // Skip target image
        if (targetFilename == dbFilename) {
            continue;
        }
        
        float dist = computeDistance(targetFeatures, database[i], distanceMetric);
        
        if (dist < 0.0f) {
            printf("ERROR: Distance computation failed\n");
            return -1;
        }
        
        Match m;
        m.filename = filenames[i];
        m.distance = dist;
        matches.push_back(m);
    }
    
    // Sort by distance
    if (matchMode == MatchMode::TOP) {
        std::sort(matches.begin(), matches.end(), 
                  [](const Match &a, const Match &b) {
                      return a.distance < b.distance;
                  });
    } else {
        std::sort(matches.begin(), matches.end(), 
                  [](const Match &a, const Match &b) {
                      return a.distance > b.distance;
                  });
    }
    
    // Display results
    if (matchMode == MatchMode::TOP) {
        printf("=== Top %d Matches ===\n", numMatches);
    } else {
        printf("=== Bottom %d Matches ===\n", numMatches);
    }
    
    for (int i = 0; i < numMatches && i < matches.size(); i++) {
        std::string filename = getFilename(matches[i].filename);
        printf("%d. %s\n", i + 1, filename.c_str());
    }
    
    return 0;
}