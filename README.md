# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.


[//]: # (Image References)

[image1]: ./examples/origin_center_2016_12_01_13_31_13_686.jpg 
[image2]: ./examples/origin_right_2016_12_01_13_31_13_686.jpg 
[image3]: ./examples/origin_left_2016_12_01_13_31_13_686.jpg 

[image4]: ./examples/preprocessd_center_2016_12_01_13_31_13_686.jpg 
[image5]: ./examples/preprocessd_right_2016_12_01_13_31_13_686.jpg 
[image6]: ./examples/preprocessd_left_2016_12_01_13_31_13_686.jpg 

[image7]: ./examples/argument_center_2016_12_01_13_31_13_686.jpg
[image8]: ./examples/argument_right_2016_12_01_13_31_13_686.jpg
[image9]: ./examples/argument_left_2016_12_01_13_31_13_686.jpg

[image10]: ./examples/flip_center_2016_12_01_13_31_13_686.jpg
[image11]: ./examples/flip_right_2016_12_01_13_31_13_686.jpg
[image12]: ./examples/flip_left_2016_12_01_13_31_13_686.jpg

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


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

The network that I used is inspired by the NVIDIA model 

![](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

the lambda_1 layer accept image size of 66x200 and the image is normalized ((image data divided by 127.5 and subtracted 1.0))

but I add a Drop layer to with drop prob 0.5 after the last Convolutional layer to reduce the overfit.

```python
model.add(Dropout(0.5))
```



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting

I gather the data by driving 4 circles. 2 for clockwise and 2 for anti-clockwise. 

I also take some method for generalizing as follows:
- 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

Because the simulator is very lag , so I can not gather more data by driving the car. But I apply some random jitter to the image that the course provide to overcome the overfit.

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

The final model architecture is as follows (printed by using `print(model.summary())`)



```sh
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
lambda_1 (Lambda)            (None, 66, 200, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 98, 24)        1824
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 47, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 22, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 20, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 18, 64)         36928
_________________________________________________________________
dropout_1 (Dropout)          (None, 1, 18, 64)         0
_________________________________________________________________
flatten_1 (Flatten)          (None, 1152)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               115300
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

I use images taken from the 3 cameras fixed on center,right and left.

![alt text][image1]
![alt text][image2]
![alt text][image3]

Then I do a pre-process for all these 3 images by crop and resize them to 60x200

![alt text][image4]
![alt text][image5]
![alt text][image6]

To save memory and get more data,I crafted a generator as below:

```python
def generate_batch_image(samples, batch_size=BATCH_SIZE):
    while True:
        shuffle(samples)
        for offset in range(0, len(samples), batch_size):
            batch_samples = samples[offset:offset + batch_size]
            _images = []
            _measurements = []
            for line in batch_samples:
                for i in range(3):
                    path = PATH + (line[i].strip())
                    image = imread(path)
                    image = preprocess_image(image)
                    
                    if i == 1:
                        measurement = float(line[3]) + SHEAR_CORRECTION
                    elif i == 2:
                        measurement = float(line[3]) - SHEAR_CORRECTION
                    else:
                        measurement = float(line[3])


                    trans_image,trans_measurement = argument(image,measurement)

                    _images.append(image)
                    _measurements.append(measurement)
                    _images.append(np.fliplr(image))
                    _measurements.append(-measurement)
                    _images.append(trans_image)
                    _measurements.append(trans_measurement)


            X_train = np.array(_images)
            y_train = np.array(_measurements)

            yield shuffle(X_train, y_train)
```

In the generator , I argumented the image data by:

- flip the image
- adding translation and random brightness

![alt text][image10]
![alt text][image11]
![alt text][image12]

![alt text][image7]
![alt text][image8]
![alt text][image9]




I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer and set the learning rate to 10e-4.

