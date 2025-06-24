# Underwater Image Processing for Live Camera Feed

## Overview

This Python script, `uav_live_navigation.py`, processes live camera feeds to enhance underwater images by correcting color casts and improving visibility. It uses advanced image processing techniques such as guided filtering, color compensation, and global stretching to produce clear and vibrant images, suitable for underwater navigation or exploration, particularly for Unmanned Aerial Vehicles (UAVs) or underwater vehicles.

### Key Features

* **Color Correction:** Compensates for red and blue channel attenuation.
* **Depth Estimation:** Models light attenuation to enhance transmission map accuracy.
* **Transmission Refinement:** Guided filter used to preserve edges during refinement.
* **Image Enhancement:** Global stretching in RGB and LAB spaces to boost contrast and brightness.
* **Live Processing:** Real-time video feed processing from a connected camera.

## Requirements

* Python 3.6+
* Libraries:

  * OpenCV (`opencv-python`)
  * NumPy (`numpy`)
  * SciPy (`scipy`)
  * scikit-image (`scikit-image`)

### Installation

1. **Install Python:** Download from [python.org](https://www.python.org/)
2. **Install Dependencies:**

   ```bash
   pip install opencv-python numpy scipy scikit-image
   ```
3. **Verify Installation:**

   ```bash
   python -c "import cv2, numpy, scipy, skimage; print('All libraries installed successfully')"
   ```

## Usage

1. **Save the Script:** Name the file `uav_live_navigation.py`.
2. **Connect a Camera:** Ensure a working camera is connected.
3. **Run the Script:**

   ```bash
   python uav_live_navigation.py
   ```
4. **View Output:** A window titled *"Underwater Navigation - Live Feed"* will show the enhanced feed.
5. **Exit:** Press `Ctrl+C` in terminal to stop the script.

## Configuration

The script includes a configurable `CONFIG` dictionary:

```python
CONFIG = {
    "block_size": 9,
    "gimfilt_radius": 30,
    "eps": 0.01,
    "rb_compensation_flag": 0,
    "enhancement_strength": 0.6
}
```

* `block_size`: Block size for processing.
* `gimfilt_radius`: Guided filter radius.
* `eps`: Epsilon for guided filter.
* `rb_compensation_flag`: Red/blue channel compensation toggle.
* `enhancement_strength`: LAB enhancement intensity.

## Script Details

### Classes and Functions

* **`GuidedFilter`**: Edge-preserving transmission refinement using box filters.
* **`ColourCorrection`**:

  * Red/Blue compensation
  * Depth estimation
  * Background light calculation
  * Transmission estimation and refinement
  * Radiance recovery
* **Enhancement Functions**:

  * `basic_stretching`
  * `lab_stretching`
  * `global_stretching_advanced`
  * `image_enhancement`: Combines enhancement techniques.

### Main Loop

1. Capture frame
2. Apply color correction
3. Estimate and refine depth/transmission
4. Recover radiance
5. Enhance image
6. Display output
7. Repeat

## Troubleshooting

* **Camera Not Found**: Check camera connection or change `cv2.VideoCapture(0)` to another index.
* **Frame Capture Failure**: Ensure the camera is not used by another application.
* **Performance Issues**: Lower `gimfilt_radius` or `block_size`.
* **Import Errors**: Reinstall missing libraries.

## Notes

* Tuned for underwater scenes; may be adapted for other low-visibility environments.
* Static image support: Replace camera input with `cv2.imread`.
* Assumes RGB camera feed.