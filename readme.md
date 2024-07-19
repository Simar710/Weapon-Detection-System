For detailed explanation, graphs , comparisons and study, please refer to the pdf file.

## Summary:
- Spearheaded the development of a robust multi-class weapon detection and classification system, using advanced machine learning techniques. 
- Integrated diverse weapon image datasets, over 7000 images, and employed two distinct classifiers: Feed Forward Neural Network and YOLOv8.
- Utilized Python for script development, incorporating Keras for neural network training and evaluation, and OpenCV for data preprocessing. 
- Optimized model using 5-Fold cross-validation for neural network training. 
- Trained YOLOv8 on the entire dataset, enabling real-time weapon detection in both images and videos, with organization of training setups and directory structures.

# Neural Network Training and Evaluation with K-Fold Cross-Validation
This Python script demonstrates the training and evaluation of a feed forward neural network classifier using 5-Fold cross-validation. The algorithm uses 1 hidden layer with 40 nodes hidden nodes and a learning rate of 0.03 in 30 epochs. The data is truncated to only 500 images due to memory size limitations when training the model on the entire dataset of 7000 images.

### Usage:
To retrain the model, you can simply run the ipynb file and it will install all dependencies and train the model on the images in relative path “./full_dataset”. All models are saved into a models array and can be loaded for use using the keras.models.load_model function for further analysis or making predictions.

# YOLO(you only look once) Training and Evaluation
This Python script demonstrates the training and evaluation of dataset that we made by combining multiple datasets. All the 7000 images are trained and a Yolov8 model is created. 

### Usage:
To train the model, you can simply run the ipynb file and it will install all dependencies and train the model on the images in relative path “./full_dataset”. While running the code the images and labels folder will split into train and valid folder with images and labels folders. The data.yaml file have path of train and valid images files with all the classes and the model with be trained and the location will be known in the end. For me it was runs/detect/train and to predict the images of video you must use best.pt inside train7 folder. The code is well commented, and all the steps are mentioned.

