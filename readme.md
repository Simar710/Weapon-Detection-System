# Weapon Detection System

## Project Overview

This project is a **comprehensive multi-class weapon detection and classification system** designed to identify and classify various types of weapons in images and videos using advanced machine learning and computer vision techniques. The system employs two different approaches—a Feed Forward Neural Network (FNN) and YOLOv8 object detection—to achieve robust weapon detection capabilities for security and surveillance applications.

### Key Features
- Multi-class weapon detection supporting 12 weapon categories
- Dual-model approach: Traditional neural network and state-of-the-art object detection
- Real-time weapon detection in images and videos
- Comprehensive dataset of over 7000 weapon images
- Model evaluation using cross-validation and standard metrics

---

## Tech Stack

### Programming Language
- **Python 3.x** - Core language for all implementations

### Deep Learning Frameworks
- **TensorFlow 2.14.0** - Backend framework for neural network implementation
- **Keras 2.14.0** - High-level API for building and training the Feed Forward Neural Network
- **PyTorch 2.1.0** - Backend for YOLOv8 model
- **Ultralytics** - YOLOv8 implementation library for object detection

### Data Processing & Computer Vision
- **NumPy** - Numerical computations and array operations
- **Pandas** - Data manipulation and CSV handling
- **OpenCV (cv2)** - Image preprocessing, video processing, and visualization
- **Pillow** - Image loading and manipulation

### Machine Learning & Evaluation
- **scikit-learn** - Data splitting, K-Fold cross-validation, label encoding, and metrics
- **Matplotlib** - Plotting training curves and visualizations
- **Seaborn** - Enhanced statistical visualizations

### Additional Libraries
- **SciPy** - Scientific computing operations
- **TQDM** - Progress bars for training loops
- **PyYAML** - YAML configuration file handling

---

## Project Structure

```
Weapon-Detection-System/
├── FNN/                          # Feed Forward Neural Network implementation
│   ├── GroupAssignment.ipynb     # FNN training notebook
│   ├── full_dataset/             # Dataset for FNN training (500 images)
│   └── model_fold_*.h5           # Trained FNN models from K-Fold CV
├── Yolov8/                       # YOLOv8 implementation
│   └── project/
│       ├── yolo.ipynb            # YOLOv8 training notebook
│       ├── full_dataset/         # Complete dataset (7000+ images)
│       │   ├── images/           # Training images
│       │   ├── labels/           # YOLO format annotations
│       │   ├── train/            # Training split
│       │   ├── valid/            # Validation split
│       │   └── dataset.yaml      # Dataset configuration
│       └── runs/detect/train*/   # Training outputs and model weights
├── GraphsOrResults/              # Visualization outputs
│   ├── FFNN/                     # FNN training graphs
│   └── YOLO/                     # YOLO training metrics and confusion matrices
├── Final Report.pdf              # Detailed project documentation
└── readme.md                     # This file
```

---

## Algorithms and Concepts

### 1. Feed Forward Neural Network (FNN)

#### Architecture
The FNN classifier is a traditional multilayer perceptron designed for weapon classification:

- **Input Layer**: Flattened image tensor (416 × 416 × 3 = 519,168 input features)
- **Hidden Layer**: 1 hidden layer with 40 neurons
- **Activation Function**: Sigmoid activation
- **Output Layer**: 12 neurons with softmax activation (one per weapon class)
- **Optimizer**: Stochastic Gradient Descent (SGD)

#### Hyperparameters
- **Learning Rate**: 0.03
- **Epochs**: 30
- **Batch Size**: 200
- **Image Size**: 416 × 416 × 3 (RGB)
- **Training Samples**: 500 images (due to memory constraints)

#### Training Strategy
- **K-Fold Cross-Validation**: 5-fold CV for robust model evaluation
- **Data Preprocessing**: 
  - Images resized to 416×416 pixels
  - Pixel normalization to [0, 1] range
  - Label encoding for categorical classes
  - One-hot encoding for output labels

#### Key Concepts
- **Backpropagation**: Gradient descent for weight optimization
- **Cross-Entropy Loss**: Categorical cross-entropy for multi-class classification
- **Model Persistence**: Models saved as `.h5` files for each fold

### 2. YOLOv8 (You Only Look Once v8)

#### Algorithm Overview
YOLOv8 is a state-of-the-art, single-stage object detection algorithm that treats detection as a regression problem:

- **Architecture**: Convolutional neural network with backbone, neck, and head components
- **Model Variant**: YOLOv8n (nano) - lightweight and fast version
- **Detection Method**: Direct prediction of bounding boxes and class probabilities
- **Grid-Based Approach**: Divides image into grid cells, each responsible for detecting objects

#### Key Features
- **Real-Time Detection**: Single forward pass through the network
- **Anchor-Free Design**: Direct prediction without anchor boxes
- **Multi-Scale Detection**: Detects objects at different scales
- **Class Confidence Scores**: Provides probability scores for each detection

#### Training Configuration
- **Epochs**: 5 (configurable for longer training)
- **Dataset Split**: 80% training, 20% validation
- **Data Format**: YOLO format (normalized bounding box coordinates)
- **Classes**: 12 weapon types
- **Image Augmentation**: Built-in augmentations in YOLOv8

#### YOLO Label Format
Each `.txt` label file contains annotations in the format:
```
<class_id> <x_center> <y_center> <width> <height>
```
All values are normalized to [0, 1] relative to image dimensions.

### 3. K-Fold Cross-Validation

A robust evaluation technique used in the FNN implementation:

- **Purpose**: Assess model generalization and reduce overfitting
- **Number of Folds**: 5
- **Process**: Dataset split into 5 subsets; training performed 5 times, each time using a different fold as validation
- **Benefits**: Provides more reliable performance estimates and reduces variance

### 4. Object Detection Concepts

#### Bounding Box Detection
- **Coordinates**: (x1, y1, x2, y2) format for box corners
- **IoU (Intersection over Union)**: Metric for box overlap evaluation
- **Confidence Score**: Probability that an object exists in the box
- **Non-Maximum Suppression (NMS)**: Removes duplicate detections

#### Classification vs Detection
- **Classification (FNN)**: Identifies weapon type in entire image
- **Detection (YOLO)**: Locates and classifies multiple weapons with bounding boxes

### 5. Computer Vision Preprocessing

#### Image Processing Pipeline
1. **Loading**: Reading images using OpenCV/Pillow
2. **Resizing**: Standardizing to 416×416 pixels
3. **Normalization**: Scaling pixel values to [0, 1]
4. **Color Space**: RGB format maintained
5. **Data Augmentation**: (YOLOv8 includes automatic augmentations)

---

## Dataset Information

### Weapon Classes (12 Total)
1. Rifle
2. Knife
3. Handgun
4. Axe (ax)
5. Sniper
6. Pistol
7. Shotgun
8. Spear
9. Eto (traditional weapon)
10. Cutter
11. Cleaver
12. Explosive

### Dataset Statistics
- **Total Images**: ~7000+ images
- **FNN Training**: 500 images (memory-limited subset)
- **YOLOv8 Training**: Full dataset (7000+ images)
- **Data Split**: 80% training, 20% validation
- **Source**: Combined multiple weapon image datasets

### Annotation Format
- **FNN**: Image classification (single label per image)
- **YOLOv8**: Bounding box annotations in YOLO format

---

## Usage

### Feed Forward Neural Network

#### Training
To retrain the FNN model:
```bash
cd FNN
jupyter notebook GroupAssignment.ipynb
```

The notebook will:
1. Install all required dependencies
2. Load images from `./full_dataset`
3. Perform 5-fold cross-validation
4. Save models as `model_fold_*.h5`

#### Loading Trained Models
```python
from tensorflow import keras
model = keras.models.load_model('FNN/model_fold_2.h5')
```

### YOLOv8 Object Detection

#### Training
To train YOLOv8:
```bash
cd Yolov8/project
jupyter notebook yolo.ipynb
```

The notebook will:
1. Install ultralytics library
2. Split dataset into train/validation
3. Train YOLOv8 model
4. Save weights to `runs/detect/train*/weights/best.pt`

#### Image Prediction
```python
from ultralytics import YOLO

model = YOLO('runs/detect/train7/weights/best.pt')
results = model.predict('path/to/image.jpg')
```

#### Video Detection
```python
from ultralytics import YOLO
import cv2

model = YOLO('runs/detect/train7/weights/best.pt')
results = model.track(source="video.mp4", show=True)
```

---

## Model Performance

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Class-wise prediction analysis

### Visualization Outputs
- **FNN**: Training/validation accuracy and loss curves
- **YOLOv8**: 
  - Precision-Recall curves
  - F1 curves
  - Confusion matrices
  - Label distribution histograms
  - Sample predictions with bounding boxes

---

## Key Algorithms Summary

| Algorithm | Type | Purpose | Key Feature |
|-----------|------|---------|-------------|
| **Feed Forward Neural Network** | Classification | Identify weapon type in image | Multilayer perceptron with 5-fold CV |
| **YOLOv8** | Object Detection | Locate and classify weapons | Real-time detection with bounding boxes |
| **K-Fold Cross-Validation** | Evaluation | Model validation | Robust performance estimation |
| **Stochastic Gradient Descent** | Optimization | FNN weight updates | Iterative optimization with learning rate |
| **Backpropagation** | Training | Compute gradients | Error propagation for weight adjustment |
| **Non-Maximum Suppression** | Post-processing | Remove duplicate detections | Keep highest confidence boxes |

---

## Detailed Documentation

For comprehensive explanations, graphs, comparisons, and in-depth analysis, please refer to the **Final Report.pdf** file included in this repository.

---

## Future Enhancements
- Expand FNN training to full dataset with optimized memory management
- Implement ensemble methods combining FNN and YOLO predictions
- Add more weapon categories and edge cases
- Deploy as web service or mobile application
- Integrate with live camera feeds for real-time surveillance
- Improve model accuracy with transfer learning and data augmentation

---

## License
This project is for educational and research purposes.
