# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The network that I used is inspired by the NVIDIA model and add dropout layer to reduce overfitting.

the lambda_1 layer accept image size of 64x64 and the image is normalized ((image data divided by 127.5 and subtracted 1.0))



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)


|Layer (type) |                Output Shape|              Param |
|:--:|:--:|:--:|
|conv2d_1 (Conv2D)            |(None, 32, 32, 24)  |      1824|
|activation_1 (Activation)   | (None, 32, 32, 24)      |  0|
|max_pooling2d_1 (MaxPooling2 |(None, 31, 31, 24)      |  0|
|conv2d_2 (Conv2D)       |     (None, 16, 16, 36)      |  21636|
|activation_2 (Activation)   | (None, 16, 16, 36)     |   0|
|max_pooling2d_2 (MaxPooling2| (None, 15, 15, 36)  |      0|
|conv2d_3 (Conv2D)      |      (None, 8, 8, 48)   |       43248|
|activation_3 (Activation)  |  (None, 8, 8, 48)     |     0|
|max_pooling2d_3 (MaxPooling2| (None, 7, 7, 48)     |     0|
|conv2d_4 (Conv2D)       |     (None, 7, 7, 64)  |        27712|
|activation_4 (Activation)  |  (None, 7, 7, 64)     |     0|
|max_pooling2d_4 (MaxPooling2| (None, 6, 6, 64)   |       0|
|conv2d_5 (Conv2D)          |  (None, 6, 6, 64)         | 36928|
|activation_5 (Activation)   | (None, 6, 6, 64)        |  0|
|max_pooling2d_5 (MaxPooling2| (None, 5, 5, 64)       |   0|
|dropout    |    (None, 5, 5, 64)    |       0|
|flatten_1 (Flatten)     |     (None, 1600)       |       0|
|dense_1 (Dense)      |        (None, 1164)      |        1863564|
|dense_2 (Dense)      |        (None, 100)       |        116500|
|dense_3 (Dense)       |       (None, 50)        |        5050|
|dense_4 (Dense)      |        (None, 10)       |         510|
|dense_5 (Dense)      |        (None, 1)         |        11|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.