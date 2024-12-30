# License plate detection using YOLO

This repository demonstrates the use of YOLOv8 for training a model on a car license plate detection dataset. It includes code for training, analyzing training metrics such as losses and mAP, and visualizing results.

## What is YOLO?

YOLO (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. It predicts bounding boxes and class probabilities directly from full images in a single evaluation, making it exceptionally fast and efficient.

### Advantages of YOLO
- **Real-time Performance**: Optimized for speed and accuracy, suitable for applications requiring real-time object detection.
- **Single-Pass Detection**: YOLO processes the entire image in one pass, reducing computational overhead.
- **Generalization**: Performs well across various datasets and environments.

### What’s New in YOLOv8?
- **Improved Architectures**: Enhanced backbone and neck designs for better feature extraction and fusion.
- **Task-Specific Models**: Separate models for detection, segmentation, and classification tasks.
- **Ease of Use**: Simplified APIs for training and deploying models.
- **Better Performance**: Higher mAP and lower latency compared to earlier versions.

---
### Table summarizing each loss and metric
| **Name**                 | **Explanation**                                                                                       | **Objective**                                      |
|--------------------------|-------------------------------------------------------------------------------------------------------|---------------------------------------------------|
| **`train/box_loss`**     | Measures the error in predicted bounding box coordinates relative to ground truth (e.g., IoU loss).   | Improve bounding box accuracy.                   |
| **`train/cls_loss`**     | Measures the error in classifying the object inside a bounding box.                                   | Improve classification accuracy.                 |
| **`train/dfl_loss`**     | Distribution Focal Loss; improves bounding box regression by modeling coordinate distributions.       | Enhance precision in bounding box localization.  |
| **`metrics/precision(B)`** | Precision: Fraction of correct positive detections out of all predicted positives.                   | Avoid false positives.                           |
| **`metrics/recall(B)`**  | Recall: Fraction of correct positive detections out of all actual positives.                         | Avoid missing objects (false negatives).         |
| **`metrics/mAP50(B)`**   | Mean Average Precision at 50% IoU threshold.                                                        | Measure detection accuracy at standard IoU.      |
| **`metrics/mAP50-95(B)`**| Mean Average Precision averaged across IoU thresholds (50%-95%, step 5%).                            | Measure comprehensive detection accuracy.        |


## Libraries Used

- **os**: Used for interacting with the file system to manage directories and files.
- **pandas**: Provides data manipulation and analysis tools, used for reading and processing `results.csv`.
- **matplotlib**: A plotting library for visualizing training metrics such as loss and mAP.
- **ultralytics**: The official Python package for YOLO models, used for training, validation, and inference.
- **streamlit**: An interactive web application framework, used to create a user-friendly interface for analyzing results and visualizations

---

## Dataset

The dataset for car license plate detection can be downloaded from Kaggle:
[Car Plate Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download)

---
## Clone the Repository
First, clone this repository to your local machine:
   ```bash
   git clone https://github.com/hit93/License_plate_detection.git
   cd yolov8-license-plate-detection
   ```
## Setting Up the Environment and Launching the App

1. **Install Python**
   - Make sure Python 3.7 or later is installed on your system. You can download it from [python.org](https://www.python.org/).

2. **Create a Virtual Environment**
   ```bash
   python -m venv yolov8-env

3. **Activate the Virtual Environment**
   ```bash
   yolov8-env\Scripts\activate
   
5. **install dependencies**
   ```bash
   pip install -r requirements.txt
   
7. **run streamlit app**
   ```bash
   streamlit run app.py
   
8. **open url http://localhost:8501/**

9. **upload image and the app will automatically detect the license plate in the image**

### Demo

![alt text](https://github.com/hit93/License_plate_detection/blob/main/demo.png)
   
