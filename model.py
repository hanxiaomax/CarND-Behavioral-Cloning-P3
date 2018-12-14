import csv
import cv2
import numpy as np
import random
import pandas as pd
import sklearn
import scipy.misc
from scipy.ndimage import rotate
from scipy.stats import bernoulli

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, MaxPooling2D, Activation
from keras.layers import Cropping2D
from keras.optimizers import Adam

STEERING_OFFSET = 0.22

samples = []
with open('/opt/carnd_p3/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for index,line in enumerate(reader):
        if index ==0:
            continue
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print("train_samples: ",len(train_samples), "validation_samples: ",len(validation_samples))

def crop(image, top_percent, bottom_percent):
    top = int(np.ceil(image.shape[0] * top_percent))
    bottom = image.shape[0] - int(np.ceil(image.shape[0] * bottom_percent))
    return image[top:bottom, :]

def random_shear(image, steering_angle, shear_range=200):

    rows, cols, ch = image.shape
    dx = np.random.randint(-shear_range, shear_range + 1)
    random_point = [cols / 2 + dx, rows / 2]
    pts1 = np.float32([[0, rows], [cols, rows], [cols / 2, rows / 2]])
    pts2 = np.float32([[0, rows], [cols, rows], random_point])
    dsteering = dx / (rows / 2) * 360 / (2 * np.pi * 25.0) / 6.0
    M = cv2.getAffineTransform(pts1, pts2)
    image = cv2.warpAffine(image, M, (cols, rows), borderMode=1)
    steering_angle += dsteering

    return image, steering_angle

def random_flip(image, steering_angle, flipping_prob=0.5):

    head = bernoulli.rvs(flipping_prob)
    if head:
        return np.fliplr(image), -1 * steering_angle
    else:
        return image, steering_angle

def random_gamma(image):

    gamma = np.random.uniform(0.4, 1.5)
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def resize(image, new_dim):

    return scipy.misc.imresize(image, new_dim)

def generate_new_image(image, steering_angle, top_crop_percent=0.35, bottom_crop_percent=0.1,
                       resize_dim=(64, 64), do_shear_prob=0.9):

    head = bernoulli.rvs(do_shear_prob)
    if head == 1:
        image, steering_angle = random_shear(image, steering_angle)

    image = crop(image, top_crop_percent, bottom_crop_percent)

    image, steering_angle = random_flip(image, steering_angle)

    image = random_gamma(image)

    image = resize(image, resize_dim)

    return image, steering_angle


# -----------------------------------------------------------------------
def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        random.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for rnd_index in range(3):
                #rnd_index = np.random.randint(0, 3)
                    name = './data/IMG/'+batch_sample[rnd_index].split('/')[-1]
                    image = cv2.imread(name)
                    if rnd_index == 1:
                        angle = float(batch_sample[3])+STEERING_OFFSET
                    elif rnd_index == 2:
                        angle = float(batch_sample[3])-STEERING_OFFSET
                    else:
                        angle = float(batch_sample[3])

                    new_image, new_angle = generate_new_image(image,angle)
                    images.append(new_image)
                    angles.append(new_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)
        
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 127.5 - 1, input_shape=(64, 64, 3)))

model.add(Convolution2D(24, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(36, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(48, 5, 5, border_mode='same', subsample=(2, 2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))

model.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1)))
model.add(Dropout(0.8))
model.add(Flatten())
model.add(Dense(1164))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(loss='mse', optimizer=Adam(1e-4))
#model.fit(np.array(x_train), np.array(y_train), validation_split=0.2, shuffle=True, epochs=8)

# model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, 
#             nb_val_samples=len(validation_samples), epoch=3)

model.fit_generator(train_generator, steps_per_epoch=len(train_samples)/32, epochs=3, verbose=1, callbacks=None, validation_data=validation_generator, validation_steps=len(validation_samples)/32)


model.save('model-test.h5')