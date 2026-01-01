# Tech Stack, Tools, and Concepts

This document provides a comprehensive overview of all technologies, tools, libraries, and concepts used in the Weapon Detection System project.

---

## Table of Contents
- [Programming Languages](#programming-languages)
- [Machine Learning Frameworks](#machine-learning-frameworks)
- [Deep Learning Libraries](#deep-learning-libraries)
- [Computer Vision Libraries](#computer-vision-libraries)
- [Data Processing and Scientific Computing](#data-processing-and-scientific-computing)
- [Model Architectures](#model-architectures)
- [Key Concepts](#key-concepts)
- [Development Environment](#development-environment)
- [Dataset](#dataset)

---

## Programming Languages

### Python 3.9+
- **Purpose**: Primary programming language for the entire project
- **Usage**: Used for data preprocessing, model training, evaluation, and inference

---

## Machine Learning Frameworks

### TensorFlow 2.14.0
- **Purpose**: Deep learning framework for building and training neural networks
- **Components Used**:
  - TensorFlow Core for computation
  - TensorBoard for visualization and monitoring
  - TensorFlow Estimator for high-level model training
  - TensorFlow I/O for filesystem operations

### Keras 2.14.0
- **Purpose**: High-level neural networks API (integrated with TensorFlow)
- **Usage**: 
  - Building Feed Forward Neural Network (FNN) architecture
  - Model training and evaluation
  - Model serialization (saving/loading .h5 files)

### PyTorch 2.1.0+
- **Purpose**: Deep learning framework used by YOLOv8
- **Components**:
  - PyTorch Core for tensor operations
  - TorchVision 0.16.0+ for computer vision utilities

### Ultralytics YOLOv8
- **Version**: 8.0.219+
- **Purpose**: State-of-the-art object detection framework
- **Model Used**: YOLOv8n (nano variant)
- **Features**:
  - Real-time object detection
  - Multi-class weapon classification
  - Support for image and video inference

---

## Deep Learning Libraries

### scikit-learn 1.3.1+
- **Purpose**: Machine learning utilities and evaluation metrics
- **Components Used**:
  - `train_test_split`: Data splitting for train/validation sets
  - `preprocessing`: Data normalization and encoding
  - Cross-validation utilities
  - Model evaluation metrics

---

## Computer Vision Libraries

### OpenCV (opencv-python) 4.8.1+
- **Purpose**: Image processing and computer vision operations
- **Usage**:
  - Image loading and preprocessing
  - Image resizing and normalization
  - Video processing for real-time detection
  - Data augmentation

---

## Data Processing and Scientific Computing

### NumPy 1.26.0+
- **Purpose**: Numerical computing and array operations
- **Usage**:
  - Array manipulations
  - Mathematical operations
  - Data preprocessing

### Pandas 2.1.1+
- **Purpose**: Data manipulation and analysis
- **Usage**:
  - CSV file reading and processing
  - Dataset management
  - Label file processing

### Matplotlib 3.8.0+
- **Purpose**: Data visualization
- **Usage**:
  - Plotting training metrics
  - Visualizing model performance
  - Creating graphs and charts

### Seaborn 0.12.2+
- **Purpose**: Statistical data visualization
- **Usage**: Enhanced visualization of results and comparisons

### SciPy 1.11.3+
- **Purpose**: Scientific computing utilities
- **Usage**: Statistical operations and scientific computations

---

## Model Architectures

### 1. Feed Forward Neural Network (FNN)
**Architecture Details**:
- **Type**: Multi-layer Perceptron
- **Input Layer**: Flattened image pixels
- **Hidden Layers**: 1 hidden layer with 40 nodes
- **Activation Function**: ReLU (Rectified Linear Unit)
- **Output Layer**: Softmax for multi-class classification
- **Learning Rate**: 0.03
- **Epochs**: 30
- **Training Strategy**: 5-Fold Cross-Validation
- **Dataset Size**: 500 images (subset due to memory constraints)

**Purpose**: Traditional neural network approach for weapon classification

### 2. YOLOv8 (You Only Look Once v8)
**Architecture Details**:
- **Model Variant**: YOLOv8n (nano - lightweight version)
- **Type**: Single-stage object detection
- **Framework**: Ultralytics implementation
- **Pretrained**: Yes (transfer learning from COCO dataset)
- **Training Epochs**: Variable (2-10 based on experiments)
- **Batch Size**: 16
- **Image Size**: 640x640 pixels
- **Dataset Size**: Full 7000+ images
- **Optimizer**: Auto (AdamW/SGD)

**Purpose**: Real-time weapon detection and localization in images and videos

---

## Key Concepts

### Machine Learning Concepts

#### 1. Multi-Class Classification
- **Description**: Classifying weapons into multiple categories
- **Classes**: 12 weapon types including rifle, knife, handgun, ax, sniper, pistol, shotgun, spear, eto, cutter, cleaver, and explosive
- **Approach**: Softmax activation for probability distribution across classes

#### 2. K-Fold Cross-Validation (5-Fold)
- **Description**: Model validation technique to assess generalization
- **Implementation**: Dataset split into 5 folds
- **Process**: Train on 4 folds, validate on 1 fold, repeat 5 times
- **Purpose**: Reduce overfitting and provide robust performance metrics

#### 3. Transfer Learning
- **Description**: Using pre-trained models as starting point
- **Implementation**: YOLOv8 pretrained on COCO dataset
- **Benefit**: Faster convergence and better performance with limited data

#### 4. Object Detection
- **Description**: Locating and classifying objects within images
- **Components**:
  - Bounding box prediction
  - Class probability prediction
  - Non-Maximum Suppression (NMS)

#### 5. Data Augmentation
- **Techniques Used** (in YOLOv8):
  - Horizontal flip (50% probability)
  - HSV color space augmentation
  - Mosaic augmentation
  - Translation and scaling
  - Mixup augmentation

### Computer Vision Concepts

#### 1. Real-Time Detection
- **Description**: Processing video frames or images with low latency
- **Implementation**: YOLOv8's single-pass architecture
- **Frame Rate**: Optimized for real-time inference

#### 2. Bounding Box Regression
- **Description**: Predicting precise object locations
- **Format**: YOLO format (x_center, y_center, width, height) normalized to image dimensions

#### 3. Confidence Scoring
- **Description**: Probability that detected object belongs to a class
- **Threshold**: IoU (Intersection over Union) threshold of 0.7
- **Purpose**: Filter low-confidence predictions

---

## Development Environment

### Jupyter Notebooks
- **Purpose**: Interactive development and experimentation
- **Files**:
  - `FNN/GroupAssignment.ipynb`: Feed Forward Neural Network implementation
  - `Yolov8/project/yolo.ipynb`: YOLOv8 training and inference

### Package Management
- **Tool**: pip (Python package installer)
- **Usage**: Installing and managing Python dependencies

### File Formats
- **Models**: 
  - `.h5` (Keras HDF5 format for FNN models)
  - `.pt` (PyTorch format for YOLOv8 weights)
- **Data**: 
  - `.csv` (labels and metadata)
  - `.yaml` (configuration files)
  - `.jpg`, `.jpeg`, `.png` (image formats)
  - `.webm` (video format for testing)
- **Labels**: `.txt` (YOLO format annotations)

---

## Dataset

### Dataset Composition
- **Total Images**: ~7,000 images
- **Source**: Combined from multiple weapon datasets
- **Split**: 
  - Training set: ~80%
  - Validation set: ~20%
- **Annotations**: YOLO format bounding boxes with class labels

### Weapon Classes (12 total)
1. Rifle
2. Knife
3. Handgun
4. Ax
5. Sniper
6. Pistol
7. Shotgun
8. Spear
9. Eto
10. Cutter
11. Cleaver
12. Explosive/Grenade

### Data Organization
- **Structure**: Separate folders for images and labels
- **Training Format**: 
  - `full_dataset/train/images/`: Training images
  - `full_dataset/train/labels/`: Training annotations
  - `full_dataset/valid/images/`: Validation images
  - `full_dataset/valid/labels/`: Validation annotations

---

## Additional Tools and Libraries

### Supporting Libraries
- **h5py**: HDF5 file operations for Keras models
- **PyYAML**: YAML file parsing for configurations
- **tqdm**: Progress bars for training loops
- **psutil**: System and process monitoring
- **thop**: Model parameter and FLOP counting
- **protobuf**: Data serialization
- **requests**: HTTP library for downloading models/datasets

### Training Optimization
- **Gradient Descent Variants**: 
  - Adam optimizer (for FNN)
  - SGD with momentum (for YOLOv8)
- **Learning Rate Scheduling**: 
  - Initial LR: 0.01 (YOLOv8)
  - Final LR: 0.01 (YOLOv8)
  - Warmup epochs: 3
- **Regularization**:
  - Weight decay: 0.0005
  - Label smoothing: 0.0

---

## Model Outputs

### FNN Output
- **Format**: Model saved as `.h5` files (one per fold)
- **Files**: `model_fold_2.h5`, `model_fold_3.h5`, `model_fold_4.h5`, `model_fold_5.h5`, `model_fold_6.h5`
- **Inference**: Class predictions with probability scores

### YOLOv8 Output
- **Format**: PyTorch `.pt` weights file
- **Location**: `runs/detect/train7/best.pt` (or similar)
- **Inference**: Bounding boxes + class labels + confidence scores
- **Visualization**: Images/videos with detected weapons highlighted

---

## Performance Metrics

### Evaluation Metrics Used
- **Classification Metrics** (FNN):
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - Confusion Matrix

- **Detection Metrics** (YOLOv8):
  - mAP (mean Average Precision)
  - Precision
  - Recall
  - IoU (Intersection over Union)

---

## References

For detailed implementation, please refer to:
- **FNN Implementation**: `FNN/GroupAssignment.ipynb`
- **YOLOv8 Implementation**: `Yolov8/project/yolo.ipynb`
- **Project Report**: `Final Report.pdf`
- **Results and Graphs**: `GraphsOrResults/` directory

---

**Note**: This project demonstrates two complementary approaches to weapon detection:
1. **FNN**: Traditional deep learning classification approach
2. **YOLOv8**: Modern real-time object detection approach

Both models are trained on the same dataset, allowing for performance comparison and analysis of different methodologies.
