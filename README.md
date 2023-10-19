# End to End Object Detection System

This project focuses on creating a End to end object detection system pipeline. in this repositories, I focused on creating face object detection model, but it can be used just for any object detection.

## Library Used

- **TensorFlow**: Python Deep Learning Framework. We will use VGG16 to build the model with keras functional API.
- **OpenCV**: Computer Vision Libraries. Used for collecting data and real-time demo.
- **LabelMe**: Manually Label/Annotate Dataset for the image acquired from opencv.
- **Albumentations**: Image augmentation Library. We will generate 60 image from each image taken and labeled. this library also augment the annotation.
- **Matplotlib**: Used to visualize the collected and augmented images and training graph.

## Notebook File

- **1-Data Collect and Label**
- **2-Data Split and Augmentation**
- **3-Model Building**
- **4-Realtime Detection**

### 1-Data Collect and Label

This notebook is designed for collecting images from a camera feed. It utilizes `OpenCV` to automatically capture an image every 0.5 seconds. The collected images are then saved as JPG files in the specified folder. The final cell of the notebook activates LabelMe, allowing for the manual labeling of objects. To understand how to use it, refer to the tutorial in the references section.

### 2-Data Split and Augmentation

This notebook has many sections explained as follows:

- **_Load image_**: this section loads the image from folder using `tf.data.Dataset`. the variables then mapped into `load_image()` function to decode the jpg into uint8 tensor.
- **_View loaded image using Matplotlib_**: generate visualization with `matplotlib` subplots for the image loaded.
- **_Splitting Data_**: On this section, split the data manually to train, test, val folder. the code inside is to move the label in original data folder into the splitted folder.
- **_Augmenting Single Data_**: This section showcase how to augment a single image and label data. It uses `albumentations` with the following augmentation rule list. it also shows how to visualize the augmentation using opencv and matplotlib.

  - RandomCrop(width=450, height=450)
  - HorizontalFlip(p=0.5)
  - RandomBrightnessContrast(p=0.2)
  - RandomGamma(p=0.2)
  - RGBShift(p=0.2)
  - VerticalFlip(p=0.5)

- **_Run Augmentation Pipeline on train dataset_**: This section loop the augmentation pipeline like previous section for large number of splitted datasets.
- **_Load Augmented File_**: This section load the complete dataset on individual variables of train, test, val using the `load_image()` function
- **_Prepare Labels_**: This section loads the label json file into individual variables of train, test, val.
- **_Combine Label and Images_**: This section combine the label and image into one row. the combined variables are then shuffled, batched, and set the prefetch value.
- **_View Image Sample_**: This section generate visualization of the finalized data.

### 3-Model Building

There are three ways of building the models in this notebook. First is to use VGG16, Second is to use VGG16 with imagenet weight, Third is to use VGG16 with untrainable imagenet layer (for feature extraction only).
This notebook has many sections explained as follows:

- **_Load Dataset_**: uses the previous notebook code to load the dataset and visualize part of it.
- **_Model Building_**: it contains `build_model()` function to construct `Keras Functional API` to produce neural network with two outputs: binary (for face), and regression (for bounding boxes). the model uses VGG as base layer then splitted into to branch to classifiy the face and calculate the bounding boxes.
- **_Defining Losses and Optimizer_**: This section define the optimizers for the model, batches/epoch, learning rate decay, and loss. The face classification uses `BinaryCrossentropy` while the bounding box regression use `localization_loss`. The `localization_loss` function is a custom loss function commonly used in object detection tasks, particularly in the context of bounding box localization. This function calculates a loss value based on the difference between the predicted bounding box coordinates and the ground truth bounding box coordinates.
  The function calculates the loss in two components:

  1. **_Delta Coordinate Loss_**: It measures the squared differences between the predicted top-left (x1, y1) coordinates and the ground truth coordinates. This component helps to penalize errors in the positioning of bounding boxes.

  2. **_Delta Size Loss_**: It calculate the squared differences in the width and height of predicted bounding boxes compared to the ground truth. This component focuses on the size of the bounding boxes.

     The final loss is the sum of these two components, which represents the overall difference between predicted and ground truth bounding boxes.

- **_Train Neural Network_**: This section define a custom Keras model class `FaceTracker`. The class serves as a wrapper around an existing model, providing custom logic for training and testing specifically for face detection tasks. It combines binary classification and localization losses to train the model and allows to easily configure and compile the model for training.
- **_Plot The Performance_**: This section uses `matplotlib` to plot the loss of the model during training.
- **_Test_**: This section can be used to test the model on test dataset.
- **_Save the Model_**: Save the model as h5 files.

### 4-Realtime Detection

This notebook uses `OpenCV` and `TensorFlow` to perform real-time face detection. it loads the model and then do video capture. The app crops the frame of the camera to a region of interest from pixel row 50 to 500 and col 50 to 500. It then converts the color space of the frame from BGR to RGB. After that, the captured frame are then resized to 120\*120 to be model input. the `FaceTracker` model then used to predict the frame input and return a sample coordinates.

on this program, I use 0.5 as a treshold (face detection confidence score) for the script to draw rectangle and place text into the object. if the confidence score is greater than 0.5, it will draw the bounding boxes using sample coordinates from the model prediction and give the text label. to exit the app use the keyword `q`.

## Reference

Tutorial: https://www.youtube.com/watch?v=N_W4EYtsa10&ab_channel=NicholasRenotte
