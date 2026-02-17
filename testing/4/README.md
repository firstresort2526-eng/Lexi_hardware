# Welcome to part 4 - Pencil AI

Side note: this readme is mainly written by deepseek cuz im too lazy lol

# Pencil Tip Detection Pipeline

This script combines SAM (Segment Anything Model) object detection (draws bounding box) with a custom CNN built with tensorflow to detect pencil tips in images.

## Overview

The pipeline:

1. Uses SAM via Roboflow to segment a pencil from an image
2. Crops the image around the detected pencil
3. Pads and resizes the crop to 480x480
4. Runs a custom neural network to predict the pencil tip coordinates
5. Converts the coordinates back to the original image space

## Dependencies

```
inference-sdk
Pillow
tensorflow
numpy
matplotlib
```

Please refer to requirements.txt for the dependencies.

## Setup

1. Install dependencies:

```bash
pip install inference-sdk Pillow tensorflow numpy matplotlib
```

2. Place your trained model (`best_model4.keras`) in the project root directory

3. Create a `images/` folder and add your test images

4. Create a `temp/` folder for intermediate files

These steps are done for you by default.

## Usage

1. Update the `IMAGE_PATH` variable in the script:

```python
IMAGE_PATH = "./images/your_image.jpg"
```

2. Run the script:

```bash
python 4.1.2_combine_detection.py
```

## How It Works

### 1. SAM Segmentation

- Resizes input image to 300x400 for SAM processing
- Queries Roboflow's SAM workflow with prompt "pencil"
- Returns bounding box coordinates for the detected pencil

### 2. Cropping & Preprocessing

- Crops the original image to the pencil bounding box
- Pads the crop to a square (480x480) with white background
- Maintains aspect ratio during padding

### 3. Tip Detection

- Loads a custom TensorFlow model with `HeatmapToCoords` layer
- Model outputs normalized coordinates (0-480) for the pencil tip
- Predictions are clamped to ensure they're within the actual pencil region (not padding)

### 4. Coordinate Conversion

- Converts 480x480 coordinates back to original image dimensions
- Accounts for padding and scaling factors
- Returns final coordinates in the original image space

## Custom Layer

The script includes a custom Keras layer `HeatmapToCoords` that:

- Takes 2-channel heatmaps (x and y predictions)
- Applies softmax to each channel
- Computes weighted average coordinates using grid sampling
- Returns normalized (x,y) coordinates

## Output

The script displays two plots:

1. The cropped 480x480 image with the predicted tip location
2. The original image with the converted tip coordinates

## File Structure

```
project/
├── 4.1.2_combine_detection.py
├── best_model4.keras
├── images/
│   └── your_image.jpg
├── temp/
│   ├── new_img.jpg
│   └── cropped.jpg
└── README.md
```

## Notes

- The script uses Roboflow's API - you'll need to replace the API key with your own
- Temporary files in `temp/` are cleaned up after execution
- The model expects input images padded to 480x480 with white background

## Troubleshooting

- **API Key Error**: Update the `api_key` in `InferenceHTTPClient`
- **Model Loading Error**: Ensure `best_model4.keras` is in the correct path
- **No Pencil Detected**: Try different images or adjust SAM prompts
