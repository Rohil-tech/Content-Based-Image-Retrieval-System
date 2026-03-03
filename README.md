# CS 5330 - PR-CV - Assignment 2: Content-Based Image Retrieval
**Student:** Rohil Kulshreshtha
February 7, 2026

**Operating System:** Windows 11
**IDE:** Visual Studio Code with CMake Tools extension
**Compiler:** MSVC 14.50 (Visual Studio 2022 Build Tools)

## Project Overview
Comprehensive content-based image retrieval (CBIR) system implementing multiple feature extraction methods, distance metrics, and an interactive web interface for finding visually similar images.

## System Architecture
Two-program design for efficient querying:
- **buildVectorDB**: Extracts features from image database, saves to CSV
- **match**: Queries pre-computed features to find similar images
- **GUI**: Gradio web interface for interactive exploration

## File Structure
```
Assignment2/
├── src/               # C++ source files
│   ├── buildVectorDB.cpp
│   ├── match.cpp
│   ├── features.cpp
│   ├── distance.cpp
│   ├── types.cpp
│   ├── csv_util.cpp
│   ├── faceDetect.cpp
│   └── filter.cpp
├── include/           # Header files
│   ├── features.h
│   ├── distance.h
│   ├── types.h
│   ├── csv_util.h
│   ├── faceDetect.h
│   ├── filter.h
│   └── dirent.h
├── gui/               # Python web interface
│   └── image_retrieval_ui.py
├── bin/               # Executables and model files
│   ├── buildVectorDB.exe
│   ├── match.exe
│   └── haarcascade_frontalface_alt2.xml
├── data/
│   ├── olympus/       # Image database
│   └── features/      # Pre-computed feature CSV files
└── build_features.bat # Automated feature extraction script
```

## Dependencies
- **C++ Backend:**
  - OpenCV 4.12.0
  - CMake 3.15+
  - C++17 compiler
  - MSVC 14.50 or compatible

- **Python GUI:**
  - Python 3.8+
  - Gradio 6.0+
  - Pillow (PIL)

## Build Instructions

### C++ Backend
```bash
# Create build directory
mkdir build
cd build

# Configure with CMake
cmake ..

# Build executables
cmake --build . --config Release
```

### Python Dependencies
```bash
pip install gradio pillow
```

## Usage

### Method 1: Command Line Interface

#### Step 1: Build Feature Database
```bash
# Single feature type
bin\buildVectorDB.exe data\olympus <feature_type> data\features\<output.csv>

# Build all features automatically in a single go
build_features.bat
```

#### Step 2: Query Images
```bash
bin\match.exe <target_image> <feature_csv> <feature_type> <distance_metric> <num_matches> [match_mode]

# Examples:
bin\match.exe data\olympus\pic.1016.jpg data\features\baseline_7x7.csv baseline_7x7 ssd 4
bin\match.exe data\olympus\pic.0164.jpg data\features\hist_rg_16.csv histogram_rg_16 intersection 3
bin\match.exe data\olympus\pic.0535.jpg data\features\texture_color_8.csv texture_color_8 texture_color 5 bottom
```

### Method 2: Web Interface (Recommended)
```bash
# Launch Gradio UI from project root
python gui\image_retrieval_ui.py

# Open browser to: http://127.0.0.1:7860
```

## Feature Types (17 Total)

### Baseline Features
- `baseline_5x5` - 5×5 center square (75 values)
- `baseline_7x7` - 7×7 center square (147 values)
- `baseline_9x9` - 9×9 center square (243 values)

### Histogram Features
- `histogram_rg_8` - RG chromaticity, 8 bins (64 values)
- `histogram_rg_16` - RG chromaticity, 16 bins (256 values)
- `histogram_rg_16_smooth` - RG with Gaussian smoothing (256 values)
- `histogram_rgb_8` - Full RGB color, 8 bins (512 values)
- `histogram_multi_rgb_8` - Top/bottom RGB histograms (1024 values)

### Texture + Color Features
- `texture_color_8` - RGB + Sobel magnitude (520 values)
- `texture_color_gabor_8` - RGB + Gabor filters (552 values)
- `texture_color_laws_8` - RGB + Laws filters (576 values)
- `texture_color_fourier_8` - RGB + Fourier spectrum (768 values)
- `texture_color_cm_8` - RGB + GLCM features (532 values)

### Advanced Features
- `deep_resnet18` - Pre-trained ResNet18 embeddings (512 values)
- `face_aware_rgb_8` - Face + background histograms (1024 values)

### Custom Features
- `custom_centered_object` - Centered composition detector (530 values)
- `custom_blue_sky` - Outdoor sky scene detector (547 values)

## Distance Metrics

### General Purpose
- `ssd` - Sum of Squared Differences (for baseline features)
- `l1` - Manhattan distance (for baseline features)
- `linf` - Chebyshev distance (for baseline features)
- `intersection` - Histogram intersection (for histogram features)
- `cosine` - Cosine distance (for deep embeddings)

### Specialized
- `multi_intersection` - Multi-histogram matching
- `texture_color` - Sobel texture + color
- `texture_color_gabor` - Gabor texture + color
- `texture_color_laws` - Laws texture + color
- `texture_color_fourier` - Fourier texture + color
- `texture_color_cm` - GLCM texture + color
- `face_aware` - Face-weighted matching
- `custom_centered_object` - Centered object distance
- `custom_blue_sky` - Sky scene distance

## Recommended Feature-Metric Pairings
- Baseline features → `ssd`, `l1`, `linf`
- Histogram features → `intersection`
- Multi-histogram → `multi_intersection`
- Texture+Color → corresponding texture metric
- Deep ResNet18 → `cosine`
- Custom features → custom distance metrics

## Command-Line Help
```bash
# View buildVectorDB options
bin\buildVectorDB.exe --help

# View match options
bin\match.exe --help
```

## Web Interface Features
- **17 feature types** with automatic metric recommendations
- **Flexible distance metric selection** with compatibility warnings
- **1-20 match results** with dynamic grid display
- **Top/Bottom mode** for finding similar or dissimilar images
- **Visual feedback** showing matched images in 5×4 grid
- **Real-time validation** with error messages and status updates

## Example Workflows

### Task 1: Baseline Matching
```bash
bin\buildVectorDB.exe data\olympus baseline_7x7 data\features\baseline_7x7.csv
bin\match.exe data\olympus\pic.1016.jpg data\features\baseline_7x7.csv baseline_7x7 ssd 4
```
Expected: pic.0986.jpg, pic.0641.jpg, pic.0547.jpg, pic.1013.jpg

### Task 2: Histogram Matching
```bash
bin\buildVectorDB.exe data\olympus histogram_rg_16 data\features\hist_rg_16.csv
bin\match.exe data\olympus\pic.0164.jpg data\features\hist_rg_16.csv histogram_rg_16 intersection 3
```

### Task 5: Deep Learning Embeddings
```bash
# ResNet18 features pre-computed, no buildVectorDB needed
bin\match.exe data\olympus\pic.0893.jpg data\features\ResNet18_olym.csv deep_resnet18 cosine 3
```

## Testing Extensions

### Extension 1: Additional Feature Types
```bash
# Test baseline variants
bin\match.exe data\olympus\pic.1016.jpg data\features\baseline_5x5.csv baseline_5x5 ssd 5
bin\match.exe data\olympus\pic.1016.jpg data\features\baseline_9x9.csv baseline_9x9 l1 5

# Test histogram variants
bin\match.exe data\olympus\pic.0164.jpg data\features\hist_rg_8.csv histogram_rg_8 intersection 3
bin\match.exe data\olympus\pic.0164.jpg data\features\histogram_rg_16_smooth.csv histogram_rg_16_smooth intersection 3

# Test texture-color combinations
bin\match.exe data\olympus\pic.0535.jpg data\features\texture_color_gabor_8.csv texture_color_gabor_8 texture_color_gabor 3
bin\match.exe data\olympus\pic.0535.jpg data\features\texture_color_laws_8.csv texture_color_laws_8 texture_color_laws 3
bin\match.exe data\olympus\pic.0535.jpg data\features\texture_color_fourier_8.csv texture_color_fourier_8 texture_color_fourier 3
bin\match.exe data\olympus\pic.0535.jpg data\features\texture_color_cm_8.csv texture_color_cm_8 texture_color_cm 3
```

### Extension 2: Face-Aware Matching
```bash
# Build face-aware features
bin\buildVectorDB.exe data\olympus face_aware_rgb_8 data\features\face_aware_rgb_8.csv

# Test with images containing faces
bin\match.exe data\olympus\pic.XXXX.jpg data\features\face_aware_rgb_8.csv face_aware_rgb_8 face_aware 5
```

### Extension 3: Web Interface
```bash
# Launch the Gradio web UI
python gui\image_retrieval_ui.py

# Interface will open at: http://127.0.0.1:7860
# Features:
# - Upload target image via drag-and-drop
# - Select from 17 feature types
# - Choose distance metric (auto-updates based on feature)
# - Adjust matches (1-20) with slider
# - Toggle between top/bottom matches
# - View results in dynamic 5×4 grid
```

### Custom Features (Task 7)
```bash
# Blue sky scene matching
bin\buildVectorDB.exe data\olympus custom_blue_sky data\features\custom_blue_sky.csv
bin\match.exe data\olympus\pic.XXXX.jpg data\features\custom_blue_sky.csv custom_blue_sky custom_blue_sky 5

# Centered object matching
bin\buildVectorDB.exe data\olympus custom_centered_object data\features\custom_centered_object.csv
bin\match.exe data\olympus\pic.XXXX.jpg data\features\custom_centered_object.csv custom_centered_object custom_centered_object 5
```

## Project Components

### Core Implementation
- Modular feature extraction system with dispatcher pattern
- Type-safe enums for features and metrics
- Efficient CSV-based feature storage and retrieval
- Self-exclusion of target image from results
- Validation of feature-metric compatibility

### Extensions
- Multiple baseline sizes (5×5, 7×7, 9×9)
- Histogram variants (8/16 bins, smoothing)
- Advanced texture analysis (Gabor, Laws, Fourier, GLCM)
- Face detection integration
- Custom domain-specific features
- Interactive web interface

## Performance Notes
- Feature extraction: ~30-60 seconds per method for 1100 images
- Matching query: <1 second per query using pre-computed features
- Gabor/Fourier: Slower extraction (~2-3 minutes) due to convolutions
- Deep features: Pre-computed, instant querying

## Troubleshooting

### Build Issues
- Ensure OpenCV is properly installed and CMAKE finds it
- Windows users: make sure dirent.h is included in `include/` folder or use a different module and modify the code accordingly.
- Place haarcascade XML in `bin/` directory
- Place csv_util.cpp and csv_util.h files in the `src/` and `include/` folders respectively

### Runtime Issues
- Verify CSV files exist in `data/features/` before matching
- Run `build_features.bat` to generate all feature databases
- Check that feature type matches the CSV file used

### GUI Issues
- Ensure Gradio 6.0+ installed: `pip install --upgrade gradio`
- Run from project root: `python gui\image_retrieval_ui.py`
- If port 7860 busy, edit `server_port` in Python file
