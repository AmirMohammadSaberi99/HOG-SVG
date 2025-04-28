# Pedestrian Detection with HOG and Haar Cascades

This repository demonstrates various approaches to detect and highlight pedestrians (and compare with Haar‑cascade detectors) using OpenCV's HOGDescriptor and a pre-trained SVM. Five scripts cover image-based detection, video processing, confidence visualization, detector comparison, and a manual sliding-window implementation.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Repository Structure](#repository-structure)
3. [Script Descriptions](#script-descriptions)
4. [Usage Examples](#usage-examples)
5. [Parameter Tuning](#parameter-tuning)
6. [License](#license)

---

## Prerequisites

- **Python 3.7+**
- **OpenCV**
  ```bash
  pip install opencv-python
  ```
- **NumPy**
  ```bash
  pip install numpy
  ```
- **Matplotlib** (for plotting/comparison scripts)
  ```bash
  pip install matplotlib
  ```

---

## Repository Structure
```plaintext
├── Practice1.py        # HOG+SVM pedestrian detection on a static image
├── Practice2.py        # HOG+SVM detection in a video file
├── Practice3.py        # Compare HOG+SVM vs. Haar full-body detector
├── Practice4.py        # Draw bounding boxes with confidence scores
└── Practice5.py        # Manual sliding-window HOG detector + NMS
```

---

## Script Descriptions

### 1. `Practice1.py`
**Static Image Detection**
- Initializes OpenCV's HOGDescriptor with the default people detector SVM.
- Runs `detectMultiScale` on a single image (`Test3.jpg`).
- Filters detections by a confidence threshold (`min_confidence = 0.6`).
- Draws green bounding boxes around detected pedestrians and displays the result.

### 2. `Practice2.py`
**Video File Detection**
- Opens a video file (`Test.mp4`) instead of a webcam.
- Performs the same HOG+SVM detection on each frame.
- Filters by confidence and overlays the pedestrian count.
- Displays annotated frames in real time; exit by pressing **q**.

### 3. `Practice3.py`
**HOG vs. Haar Cascade Comparison**
- Loads an image (`Test2.jpg`) and converts it to grayscale.
- Runs HOG+SVM pedestrian detection and Haar full-body cascade detection.
- Prints the number of detections from each method.
- Displays the original image alongside HOG (green boxes) and Haar (blue boxes) results.

### 4. `Practice4.py`
**Bounding Boxes with Confidence**
- Detects pedestrians with HOG+SVM on `Test3.jpg`.
- Draws each bounding box and overlays its raw SVM score above the box.
- Provides a visual indication of detection confidence.

### 5. `Practice5.py`
**Manual Sliding-Window HOG Detector**
- Implements an image pyramid and sliding window without `detectMultiScale`.
- Extracts HOG descriptors manually and scores them using the pre-trained SVM weights.
- Applies non-max suppression (NMS) to merge overlapping detections.
- Draws final bounding boxes and confidence scores on `Test2.jpg`.

---

## Usage Examples

1. **Static Image**
   ```bash
   python Practice1.py
   ```

2. **Video File**
   ```bash
   python Practice2.py
   ```

3. **Detector Comparison**
   ```bash
   python Practice3.py
   ```

4. **Confidence Visualization**
   ```bash
   python Practice4.py
   ```

5. **Manual Sliding-Window**
   ```bash
   python Practice5.py
   ```

*(Ensure `Test.jpg`, `Test2.jpg`, `Test3.jpg`, and `Test.mp4` are placed in the working directory or update paths accordingly.)*

---

## Parameter Tuning

Each script exposes key parameters you can tweak directly in code:

- **`winStride`**, **`padding`**, **`scale`** in `detectMultiScale`
- **`min_confidence`**: confidence threshold for filtering
- **`scaleFactor`**, **`minNeighbors`**, **`minSize`** in Haar cascades
- **Sliding-window**: `stepSize`, `pyramidScale`, `scoreThreshold`, `nmsThreshold`

Adjust these to balance detection accuracy, false positives, and performance.

---

## License

This project is released under the MIT License. See `LICENSE` for details.

