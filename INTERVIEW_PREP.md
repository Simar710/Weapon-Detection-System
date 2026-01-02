# Interview Preparation Guide - Weapon Detection System

This guide contains interview questions derived from your Weapon Detection System project, with detailed answers in simple language.

---

## Table of Contents
1. [Project Overview Questions](#project-overview-questions)
2. [Machine Learning Concepts](#machine-learning-concepts)
3. [Deep Learning & Neural Networks](#deep-learning--neural-networks)
4. [Computer Vision Questions](#computer-vision-questions)
5. [Technical Implementation](#technical-implementation)
6. [Model Comparison Questions](#model-comparison-questions)
7. [Important Terminologies](#important-terminologies)

---

## Project Overview Questions

### Q1: Can you describe your weapon detection project?

**Answer:**
I built a weapon detection system that can identify 12 different types of weapons in images and videos. The system uses two different approaches:
1. A Feed Forward Neural Network (FNN) for classification
2. YOLOv8 for real-time object detection

I worked with over 7,000 images of weapons like rifles, pistols, knives, grenades, etc. The project demonstrates both traditional neural network classification and modern object detection techniques.

**Key Points to Mention:**
- Multi-class classification (12 weapon types)
- Two different model architectures
- Real-time detection capability
- Large dataset (7,000+ images)

---

### Q2: What problem does your project solve?

**Answer:**
The project addresses security and surveillance needs by automatically detecting weapons in images or video feeds. This can be used in:
- Airport security cameras
- Public place monitoring
- Threat assessment systems
- Automated security alerts

Instead of humans watching cameras 24/7, the AI can flag suspicious items immediately.

---

## Machine Learning Concepts

### Q3: What is machine learning, and how did you use it in your project?

**Answer:**
Machine learning is when computers learn patterns from data instead of being explicitly programmed. 

**In my project:**
- I showed the computer thousands of weapon images
- The computer learned what features make each weapon type unique (shape, size, patterns)
- After training, it can identify weapons it has never seen before
- It's like teaching a child to recognize animals by showing them pictures

**The learning process:**
1. Feed the model training data (images with labels)
2. Model makes predictions
3. Compare predictions with correct answers
4. Adjust model to reduce errors
5. Repeat until it gets good at recognizing weapons

---

### Q4: What is supervised learning?

**Answer:**
Supervised learning is like learning with a teacher. You provide the computer with:
- **Input:** Images of weapons
- **Labels:** What weapon is in each image (e.g., "rifle", "pistol")

The computer learns by comparing its guesses to the correct labels you provided.

**In my project:**
Every image had a label file telling the model "this is a knife" or "this is a rifle." The model learned to map image features to these labels.

**Contrast with unsupervised learning:** That's like learning without a teacher - the computer finds patterns on its own without being told what's correct.

---

### Q5: Explain classification vs. detection in your project.

**Answer:**

**Classification (what the FNN does):**
- Looks at the whole image
- Says "This image contains a rifle"
- Answers: "What type of weapon is this?"
- No location information

**Object Detection (what YOLO does):**
- Finds where weapons are in the image
- Draws boxes around them
- Says "There's a rifle at this location and a knife at that location"
- Answers: "What weapons are here AND where are they?"

**Example:**
- Classification: "This is a picture of a gun"
- Detection: "There are 2 guns - one in the top left corner and one in the bottom right"

---

### Q6: What is K-Fold Cross-Validation, and why did you use 5-fold?

**Answer:**
K-Fold Cross-Validation is a technique to test how well your model will work on new data.

**Simple Explanation:**
Imagine you have 500 images. Instead of training on all of them once:
1. Split data into 5 equal parts (folds)
2. Train on 4 parts, test on 1 part
3. Repeat 5 times, using a different part for testing each time
4. Average the results

**Why use it:**
- Prevents overfitting (memorizing training data)
- Every image gets tested once
- More reliable performance estimate
- Uses all data efficiently

**In my project:**
I used 5-fold cross-validation for the FNN model. This gave me 5 different model versions, and I could see how consistent the performance was.

---

### Q7: What is overfitting and how did you prevent it?

**Answer:**
**Overfitting** is when your model memorizes the training data instead of learning general patterns.

**Real-world analogy:**
Like a student who memorizes answers to practice questions but can't solve new problems. They do great on practice tests but fail the real exam.

**How I prevented it:**
1. **K-Fold Cross-Validation:** Tested on data the model hadn't seen during training
2. **Train/Validation Split:** Kept 20% of data separate for testing
3. **Limited epochs:** Stopped training at 30 epochs for FNN
4. **Regularization:** Used weight decay in YOLOv8 (0.0005)

**Signs of overfitting:**
- High accuracy on training data
- Low accuracy on validation data
- Big difference between the two

---

## Deep Learning & Neural Networks

### Q8: What is a Neural Network in simple terms?

**Answer:**
A neural network is inspired by how the human brain works. It's made of layers of "neurons" that process information.

**Simple breakdown:**
1. **Input Layer:** Receives the data (like pixels from an image)
2. **Hidden Layers:** Process and find patterns (like edges, shapes, textures)
3. **Output Layer:** Makes the final decision (like "this is a rifle")

**In my FNN:**
- Input: Flattened image pixels
- Hidden Layer: 40 neurons that learn important features
- Output: 12 neurons (one for each weapon type)

**How it learns:**
Each connection between neurons has a "weight." During training, these weights adjust to recognize patterns. It's like adjusting the brightness and contrast on a photo until you can see clearly.

---

### Q9: What is a Feed Forward Neural Network (FNN)?

**Answer:**
A Feed Forward Neural Network is the simplest type of neural network where information flows in one direction: input → hidden layers → output.

**My FNN architecture:**
- **Input:** Image pixels (flattened into a 1D array)
- **Hidden Layer:** 40 neurons with ReLU activation
- **Output Layer:** 12 neurons with Softmax activation (one per weapon class)

**Why "Feed Forward"?**
Data moves forward through the network without loops or going backward during prediction. During training, errors are sent backward (backpropagation) to adjust weights.

**Limitations:**
- Treats all pixels equally (doesn't understand spatial relationships well)
- Not great for complex images
- Slower than modern architectures for images

---

### Q10: What is an activation function? What did you use?

**Answer:**
An activation function decides if a neuron should "fire" (activate) based on its input. It adds non-linearity, allowing networks to learn complex patterns.

**I used two types:**

**1. ReLU (Rectified Linear Unit) - in hidden layer:**
- Simple rule: If input is positive, keep it; if negative, make it 0
- Formula: `f(x) = max(0, x)`
- **Why:** Fast to compute, prevents vanishing gradient problem
- **Example:** Input: [-2, 3, -1, 5] → Output: [0, 3, 0, 5]

**2. Softmax - in output layer:**
- Converts numbers to probabilities that sum to 1
- **Why:** Gives probability for each weapon class
- **Example:** Output: [0.1, 0.7, 0.05, 0.15] means 70% sure it's a rifle

**Why not just use linear functions?**
Without activation functions, the neural network would just be a fancy linear regression, unable to learn complex patterns.

---

### Q11: What are hyperparameters? Which ones did you tune?

**Answer:**
Hyperparameters are settings you choose before training that control how the model learns. Unlike weights (which the model learns), you set these manually.

**My hyperparameters:**

**For FNN:**
- **Learning Rate (0.03):** How big the steps are when adjusting weights
  - Too high: Model doesn't learn properly (jumps around)
  - Too low: Training takes forever
- **Epochs (30):** How many times the model sees the entire dataset
- **Hidden Neurons (40):** How many neurons in the hidden layer
- **Batch Size:** How many images to process before updating weights

**For YOLOv8:**
- **Learning Rate (0.01):** Starting learning rate
- **Batch Size (16):** Process 16 images at once
- **Epochs (2-10):** Varies based on experiment
- **Image Size (640x640):** Resolution for training

**How to tune them:**
Trial and error, or systematic search (grid search). Monitor validation performance to find the best values.

---

### Q12: What is backpropagation?

**Answer:**
Backpropagation is how neural networks learn from mistakes. It's the process of sending errors backward through the network to adjust weights.

**Simple explanation:**
1. Make a prediction (forward pass)
2. Calculate how wrong it was (loss/error)
3. Send the error backward through the network
4. Adjust each weight based on how much it contributed to the error
5. Repeat

**Analogy:**
Like getting feedback on a test. You see which questions you got wrong and study those topics more. The network "studies" the features it got wrong.

**In my project:**
After each batch of images, backpropagation adjusted the weights to make better predictions next time. This happened thousands of times during training.

---

### Q13: What is Transfer Learning, and how did you use it?

**Answer:**
Transfer Learning is using knowledge from one task to help with another task. Instead of starting from scratch, you start with a model that already knows something useful.

**Real-world analogy:**
If you know how to ride a bicycle, learning to ride a motorcycle is easier. You transfer your balance and steering skills.

**In my project (YOLOv8):**
- Started with YOLOv8 pre-trained on COCO dataset
- COCO has 80 everyday objects (cars, people, animals)
- The model already learned to detect edges, shapes, and basic objects
- I fine-tuned it to detect weapons specifically

**Benefits:**
- Faster training (weeks → hours)
- Better accuracy with less data
- Model already understands images generally

**How it works:**
Early layers (learning edges, colors) stay mostly the same. Later layers (specific to weapons) get retrained.

---

## Computer Vision Questions

### Q14: What is Computer Vision?

**Answer:**
Computer Vision is teaching computers to "see" and understand images and videos like humans do.

**In my project:**
The computer needed to:
- Load and read images
- Understand what objects are in them
- Identify different weapon types
- Find their locations

**Common CV tasks:**
- **Image Classification:** What is this? (my FNN does this)
- **Object Detection:** What is here and where? (my YOLO does this)
- **Image Segmentation:** Outline exact object boundaries
- **Face Recognition:** Identify people

**How computers "see":**
Images are just grids of numbers (pixels). Each pixel has color values (RGB). The model learns patterns in these numbers.

---

### Q15: What is OpenCV and how did you use it?

**Answer:**
OpenCV (Open Source Computer Vision Library) is a tool that helps process images and videos.

**What I used it for:**
1. **Loading images:** `cv2.imread()` - reads image files
2. **Resizing images:** Makes all images the same size for training
3. **Color conversion:** Converting between RGB and BGR formats
4. **Preprocessing:** Normalizing pixel values (0-255 → 0-1)
5. **Video processing:** Reading video frames for real-time detection

**Why use OpenCV:**
- Fast and efficient
- Industry standard
- Works well with NumPy arrays
- Has many built-in functions for image manipulation

**Example operation:**
```python
image = cv2.imread('weapon.jpg')  # Load image
image = cv2.resize(image, (640, 640))  # Resize to 640x640
```

---

### Q16: What is Object Detection and how does YOLO work?

**Answer:**
**Object Detection** finds objects in images and draws boxes around them.

**YOLO (You Only Look Once) approach:**
Traditional methods looked at an image multiple times at different locations. YOLO looks at the entire image once, making it very fast.

**How YOLO works:**
1. Divide image into a grid (e.g., 13x13 cells)
2. Each cell predicts:
   - Bounding box coordinates (x, y, width, height)
   - Confidence score (how sure it is)
   - Class probabilities (rifle, knife, etc.)
3. Remove overlapping boxes (Non-Maximum Suppression)
4. Keep high-confidence detections

**Why I used YOLOv8:**
- Very fast (real-time detection)
- Accurate
- Can detect multiple weapons in one image
- Good for video streams

**YOLOv8n (nano):** Smallest, fastest version - perfect for balancing speed and accuracy.

---

### Q17: What is a Bounding Box?

**Answer:**
A bounding box is a rectangle that surrounds an object in an image.

**Components:**
- **x, y:** Center point coordinates
- **width, height:** Size of the box
- **Class label:** What's inside (e.g., "rifle")
- **Confidence score:** How certain the model is (0-1)

**Format I used (YOLO format):**
```
class_id x_center y_center width height
```
All values normalized (divided by image dimensions to get values 0-1).

**Example:**
```
0 0.5 0.3 0.4 0.6
```
- Class 0 (rifle)
- Center at 50% width, 30% height
- Box is 40% of image width, 60% of image height

**Why normalize:**
Makes model work with different image sizes.

---

### Q18: What is IoU (Intersection over Union)?

**Answer:**
IoU measures how much two boxes overlap. It's used to evaluate detection accuracy.

**Formula:**
```
IoU = (Area of Overlap) / (Area of Union)
```

**Visual explanation:**
- If boxes perfectly overlap: IoU = 1.0 (100%)
- If boxes don't overlap at all: IoU = 0.0 (0%)
- Usually consider detection "correct" if IoU > 0.5

**In my project:**
- Used IoU threshold of 0.7
- If predicted box and true box have IoU > 0.7, it counts as a correct detection
- Higher threshold = stricter evaluation

**Why it matters:**
A box that's slightly off-center might still be useful (high IoU), but a badly placed box (low IoU) isn't helpful.

---

### Q19: What is Data Augmentation?

**Answer:**
Data Augmentation creates new training examples by modifying existing images. It's like getting more data for free!

**Techniques used in YOLOv8:**
1. **Horizontal Flip:** Mirror the image (50% probability)
2. **Mosaic:** Combine 4 images into one
3. **HSV Augmentation:** Change brightness, saturation, hue
4. **Translation:** Shift the image left/right/up/down
5. **Scaling:** Zoom in or out slightly
6. **Mixup:** Blend two images together

**Why use it:**
- Makes model more robust
- Prevents overfitting
- Simulates different real-world conditions
- Effectively increases dataset size

**Example:**
One rifle image becomes 10 images:
- Original
- Flipped horizontally
- Slightly darker
- Slightly brighter
- Rotated a bit
- Zoomed in, etc.

---

## Technical Implementation

### Q20: What libraries did you use and why?

**Answer:**

**Deep Learning:**
- **TensorFlow/Keras:** Building and training FNN - user-friendly, industry standard
- **PyTorch:** Used by YOLOv8 - flexible, research-friendly
- **Ultralytics YOLOv8:** Pre-built object detection - saves development time

**Data Processing:**
- **NumPy:** Fast array operations, mathematical computations
- **Pandas:** Managing CSV files with labels, organizing data
- **OpenCV:** Image loading, preprocessing, resizing

**Visualization:**
- **Matplotlib:** Plotting training curves, showing results
- **Seaborn:** Statistical visualizations, prettier graphs

**Machine Learning:**
- **scikit-learn:** Train/test split, cross-validation, evaluation metrics

**Why Python:**
- Rich ML/AI ecosystem
- Easy to read and write
- Great community support
- Industry standard for ML

---

### Q21: How did you organize your dataset?

**Answer:**

**Dataset structure:**
```
full_dataset/
├── train/
│   ├── images/     (80% of data - ~5,600 images)
│   └── labels/     (annotation files)
└── valid/
    ├── images/     (20% of data - ~1,400 images)
    └── labels/     (annotation files)
```

**Label format (YOLO):**
Each image has a `.txt` file with same name:
```
0 0.5 0.5 0.3 0.4    # Class 0, center at (0.5, 0.5), size (0.3, 0.4)
```

**Classes (12 total):**
0=rifle, 1=knife, 2=handgun, 3=ax, 4=sniper, 5=pistol, 6=shotgun, 7=spear, 8=eto, 9=cutter, 10=cleaver, 11=explosive

**Why 80/20 split:**
- 80% for training (model learns from this)
- 20% for validation (tests how well it learned)
- Industry standard ratio

---

### Q22: What metrics did you use to evaluate your models?

**Answer:**

**For Classification (FNN):**
1. **Accuracy:** Percentage of correct predictions
   - Simple: (Correct predictions / Total predictions) × 100
   - Example: 90% accuracy = 90 out of 100 correct

2. **Precision:** Of all weapons I predicted as "rifle," how many were actually rifles?
   - High precision = few false alarms
   - Formula: True Positives / (True Positives + False Positives)

3. **Recall:** Of all actual rifles, how many did I detect?
   - High recall = didn't miss many
   - Formula: True Positives / (True Positives + False Negatives)

4. **F1-Score:** Balance between precision and recall
   - Harmonic mean of precision and recall
   - Good when you need both metrics to be high

**For Detection (YOLO):**
1. **mAP (mean Average Precision):** Overall detection quality across all classes
2. **IoU:** How well boxes overlap with ground truth
3. **Confidence Score:** How sure the model is about each detection

**Confusion Matrix:** Table showing which classes got confused with which.

---

### Q23: What challenges did you face with the dataset?

**Answer:**

**1. Memory Limitations:**
- Full dataset (7,000 images) was too big for FNN training
- Solution: Used only 500 images for FNN
- YOLOv8 handled all 7,000 images efficiently

**2. Class Imbalance:**
- Some weapon types had more images than others
- Could bias the model toward common classes
- Solution: Monitor per-class performance

**3. Image Quality:**
- Different resolutions, lighting conditions
- Solution: Preprocessing (resize, normalize)

**4. Annotation Quality:**
- Some label files missing or incorrect
- Solution: Data validation, skip corrupted files

**5. Real-world Variability:**
- Weapons at different angles, distances, lighting
- Solution: Data augmentation to simulate variations

---

### Q24: What is the difference between training and inference?

**Answer:**

**Training (Learning Phase):**
- Model learns from labeled data
- Adjusts weights through backpropagation
- Takes hours/days
- Uses lots of computational power
- Requires labeled dataset
- Goal: Learn patterns

**Inference (Prediction Phase):**
- Model makes predictions on new data
- Weights are frozen (no learning)
- Takes milliseconds
- Less computational power needed
- No labels needed
- Goal: Apply learned patterns

**In my project:**
- **Training:** Used 7,000 labeled images to train YOLO, took several hours
- **Inference:** Give model a new weapon image, it predicts in <1 second

**Analogy:**
- Training = studying for an exam (takes time, need textbook/answers)
- Inference = taking the exam (fast, no book needed)

---

## Model Comparison Questions

### Q25: Why did you use two different models?

**Answer:**
I wanted to compare traditional and modern approaches:

**FNN (Traditional Approach):**
- **Pros:** 
  - Simple to understand and implement
  - Good for learning fundamentals
  - Less computational requirements
- **Cons:**
  - Only does classification (what weapon)
  - Can't locate weapons in image
  - Slower and less accurate for images
  - Limited to small datasets (500 images)

**YOLOv8 (Modern Approach):**
- **Pros:**
  - Does detection (what + where)
  - Real-time performance
  - Handles full dataset (7,000 images)
  - State-of-the-art accuracy
  - Can detect multiple weapons
- **Cons:**
  - More complex
  - Requires more training data
  - Larger model size

**Which is better?**
YOLOv8 for this application because:
- Security needs location information
- Multiple weapons might be present
- Real-time detection is important

---

### Q26: What is the difference between your FNN and YOLO architectures?

**Answer:**

**FNN Architecture:**
```
Input (image pixels) 
  → Flatten to 1D array
  → Hidden Layer (40 neurons, ReLU)
  → Output Layer (12 neurons, Softmax)
  → One class prediction
```

**YOLO Architecture:**
```
Input (640×640 image)
  → Convolutional Layers (extract features)
  → Feature Pyramid Network (multi-scale detection)
  → Detection Head (predict boxes + classes)
  → Multiple predictions (boxes + classes + confidence)
```

**Key Differences:**

| Aspect | FNN | YOLO |
|--------|-----|------|
| **Input** | Flattened pixels | 2D image grid |
| **Layers** | Fully connected | Convolutional |
| **Output** | Single class | Multiple boxes + classes |
| **Spatial Info** | Lost when flattened | Preserved |
| **Speed** | Slower | Real-time |
| **Use Case** | Classification | Detection |

---

### Q27: How does YOLOv8 achieve real-time detection?

**Answer:**

**Key innovations:**

**1. Single Pass:**
- Looks at image only once
- Other methods scan image multiple times
- Much faster

**2. Efficient Architecture:**
- YOLOv8n (nano) - lightweight version
- Optimized convolutions
- Fewer parameters than larger models

**3. End-to-End:**
- One neural network does everything
- No separate region proposal step
- No post-processing pipeline

**4. GPU Optimization:**
- Designed to run on GPUs efficiently
- Parallel processing of image regions
- Batch processing

**Performance:**
- Can process 30-60 frames per second
- Suitable for live video streams
- Millisecond-level detection

**Trade-offs:**
- YOLOv8n is faster but slightly less accurate than larger variants (YOLOv8s, m, l, x)
- Chose nano for speed-accuracy balance

---

## Important Terminologies

### Q28: Explain key terms someone might ask about

**Epoch:**
One complete pass through the entire training dataset. If you have 1,000 images and train for 30 epochs, the model sees each image 30 times.

**Batch:**
Number of samples processed before updating model weights. Batch size of 16 means process 16 images, calculate average error, then update.

**Learning Rate:**
Step size when adjusting weights. Like volume control: too high and you overshoot, too low and learning is slow.

**Loss Function:**
Measures how wrong the model's predictions are. Goal of training is to minimize loss.

**Gradient:**
Direction and amount to adjust each weight. Calculated during backpropagation.

**Gradient Descent:**
Optimization algorithm that adjusts weights to minimize loss by following gradients downhill.

**Weight/Parameter:**
Numbers in the neural network that get adjusted during training. More weights = more complex model.

**Bias:**
Extra parameter in each neuron that helps shift the activation function. Like an intercept in linear regression.

**Tensor:**
Multi-dimensional array (generalization of matrices). Images are 3D tensors (height × width × color channels).

**Feature:**
Distinguishing characteristic the model learns (e.g., "rifles are long and thin").

**Stride:**
Step size when sliding a filter across an image in convolutional layers.

**Padding:**
Adding borders to images so they don't shrink after convolutions.

**Dropout:**
Randomly turning off neurons during training to prevent overfitting.

**Regularization:**
Techniques to prevent overfitting (like L1, L2 weight penalties, dropout).

---

### Q29: What is the difference between validation and test sets?

**Answer:**

**Training Set (80%):**
- Model learns from this
- Weights are adjusted based on this data
- Used during training

**Validation Set (20%):**
- Check performance during training
- Used to tune hyperparameters
- Helps detect overfitting
- Model doesn't learn from this

**Test Set (if separate):**
- Final evaluation only
- Never used during training or tuning
- True measure of real-world performance

**In my project:**
- Used 80/20 train/validation split
- For FNN, also used 5-fold cross-validation
- Each fold acted as validation set once

**Why separate them:**
If you tune hyperparameters based on test set, you're indirectly training on it. Validation set prevents this.

---

### Q30: What is normalization and why did you use it?

**Answer:**
Normalization scales data to a standard range, usually 0-1 or -1 to 1.

**Without normalization:**
- Pixel values range from 0-255
- Large numbers make training unstable
- Gradients can explode or vanish
- Slow convergence

**With normalization:**
```python
normalized_pixel = pixel_value / 255.0
# 0 → 0.0, 255 → 1.0
```

**Benefits:**
- Faster training
- More stable
- Better gradient flow
- Prevents some neurons from dominating

**Types used:**
1. **Min-Max Scaling:** Convert 0-255 → 0-1 (for images)
2. **Standardization:** Mean=0, Std=1 (for some features)

**In my project:**
Applied to image pixels before feeding to both FNN and YOLO.

---

## Tips for the Interview

### How to Talk About Your Project:

**1. Start with the Big Picture:**
"I built an AI system that can automatically detect weapons in images and videos using deep learning."

**2. Explain Your Approach:**
"I implemented two different methods - a traditional neural network and a modern YOLO detector - to compare their performance."

**3. Highlight Specific Techniques:**
"I used 5-fold cross-validation to ensure robust evaluation and transfer learning to achieve better accuracy with limited training time."

**4. Mention Results:**
"The YOLO model can detect weapons in real-time with high accuracy, making it suitable for actual security applications."

**5. Discuss Challenges:**
"One challenge was memory limitations with the full dataset, so I used a subset for the FNN while YOLO handled the complete dataset efficiently."

### Common Follow-up Questions:

**"How would you improve this?"**
- Collect more diverse data
- Try ensemble methods
- Add more data augmentation
- Deploy on edge devices for real-world use
- Add tracking across video frames

**"What did you learn?"**
- Difference between classification and detection
- Importance of data quality
- Trade-offs between accuracy and speed
- Practical ML engineering skills

**"How would you deploy this?"**
- Use a web API (Flask/FastAPI)
- Mobile app with TensorFlow Lite
- Edge device deployment
- Cloud-based processing with alerts

---

## Quick Reference: Key Numbers to Remember

- **Dataset:** 7,000+ images, 12 weapon classes
- **FNN:** 40 neurons, 0.03 learning rate, 30 epochs, 500 images
- **YOLO:** 640×640 images, batch size 16, IoU threshold 0.7
- **Split:** 80% training, 20% validation
- **Cross-validation:** 5-fold for FNN
- **Libraries:** TensorFlow 2.14, PyTorch 2.1, OpenCV 4.8, Ultralytics YOLOv8

---

## Additional Resources

If asked "What would you read to learn more?"
- YOLOv8 documentation (Ultralytics)
- TensorFlow/Keras tutorials
- Deep Learning book by Goodfellow
- Computer Vision: Algorithms and Applications
- Papers: Original YOLO paper, YOLOv3, YOLOv8

---

**Good luck with your interview! Remember:**
- Speak clearly and confidently
- Use simple language (like this guide)
- Give examples from your project
- Admit if you don't know something
- Show enthusiasm for learning
