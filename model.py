import cv2
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from keras.models import Sequential
from keras.layers import Dense, Lambda, Cropping2D, Conv2D, Dense, Activation, MaxPooling2D, Flatten,Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

BATCH_SIZE = 32
SHEAR_CORRECTION = 0.2
PATH = "/opt/carnd_p3/data/"


#read the data 
lines = []
with open(PATH+"driving_log.csv") as input:
    reader = csv.reader(input)
    for index, line in enumerate(reader):
        if index == 0:
        	continue
        lines.append(line)
    

#set trains and validations
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def preprocess_image(img):
	#crop
    new_img = img[50:140,:,:]
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    return new_img



def argument(image,steering_angle):
    image, steering_angle = random_translate(image, steering_angle,100,10)
    image = random_brightness(image)
    return image, steering_angle


def random_translate(image, steering_angle, range_x, range_y):
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, 100], [0, 1, 10]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle


def random_brightness(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


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


train_generator = generate_batch_image(train_samples, batch_size=BATCH_SIZE)
validation_generator = generate_batch_image(validation_samples, batch_size=BATCH_SIZE)

#step must divide batch size,otherwise the fit_generate will run very slow
train_steps = np.ceil(len(train_samples) / BATCH_SIZE).astype(np.int32)
validation_steps = np.ceil(len(validation_samples) / BATCH_SIZE).astype(np.int32)

model = Sequential()
#model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/127.5 - 1.0,input_shape=(66,200,3)))
#model.add(Cropping2D(cropping=((50, 25), (0, 0))))
model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(optimizer=Adam(10e-4), loss='mse')
model.fit_generator(train_generator,
                    steps_per_epoch=train_steps,
                    epochs=5,
                    validation_data=validation_generator,
                    validation_steps=validation_steps)
model.save('model2.h5')

