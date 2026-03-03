REM Rohil Kulshreshtha
REM February 2, 2026
REM CS 5330 - PR-CV - Assignment 2
REM 
REM Batch script to build all feature vector databases

@echo off

if not exist "data\features" mkdir "data\features"

echo [1/16] Building baseline_5x5...
bin\buildVectorDB.exe data\olympus baseline_5x5 data\features\baseline_5x5.csv

echo [2/16] Building baseline_7x7...
bin\buildVectorDB.exe data\olympus baseline_7x7 data\features\baseline_7x7.csv

echo [3/16] Building baseline_9x9...
bin\buildVectorDB.exe data\olympus baseline_9x9 data\features\baseline_9x9.csv

echo [4/16] Building histogram_rg_8...
bin\buildVectorDB.exe data\olympus histogram_rg_8 data\features\hist_rg_8.csv

echo [5/16] Building histogram_rg_16...
bin\buildVectorDB.exe data\olympus histogram_rg_16 data\features\hist_rg_16.csv

echo [6/16] Building histogram_rg_16_smooth...
bin\buildVectorDB.exe data\olympus histogram_rg_16_smooth data\features\histogram_rg_16_smooth.csv

echo [7/16] Building histogram_rgb_8...
bin\buildVectorDB.exe data\olympus histogram_rgb_8 data\features\hist_rgb_8.csv

echo [8/16] Building histogram_multi_rgb_8...
bin\buildVectorDB.exe data\olympus histogram_multi_rgb_8 data\features\hist_multi_rgb_8.csv

echo [9/16] Building texture_color_8...
bin\buildVectorDB.exe data\olympus texture_color_8 data\features\texture_color_8.csv

echo [10/16] Building texture_color_gabor_8...
bin\buildVectorDB.exe data\olympus texture_color_gabor_8 data\features\texture_color_gabor_8.csv

echo [11/16] Building texture_color_laws_8...
bin\buildVectorDB.exe data\olympus texture_color_laws_8 data\features\texture_color_laws_8.csv

echo [12/16] Building texture_color_fourier_8...
bin\buildVectorDB.exe data\olympus texture_color_fourier_8 data\features\texture_color_fourier_8.csv

echo [13/16] Building texture_color_cm_8...
bin\buildVectorDB.exe data\olympus texture_color_cm_8 data\features\texture_color_cm_8.csv

echo [14/16] Building custom_centered_object...
bin\buildVectorDB.exe data\olympus custom_centered_object data\features\custom_centered_object.csv

echo [15/16] Building custom_blue_sky...
bin\buildVectorDB.exe data\olympus custom_blue_sky data\features\custom_blue_sky.csv

echo [16/16] Building face_aware_rgb_8...
bin\buildVectorDB.exe data\olympus face_aware_rgb_8 data\features\face_aware_rgb_8.csv

echo.
echo Done!
pause