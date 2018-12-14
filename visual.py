import csv
import cv2
import os

import numpy as np
SHEAR_CORRECTION = 0.2
PATH = "/opt/carnd_p3/data/"
lines = []


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

def preprocess_image(img):
	#crop
    new_img = img[50:140,:,:]
    # scale to 66x200x3 (same as nVidia)
    new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
    return new_img

with open(PATH+"driving_log.csv") as input:
    reader = csv.reader(input)
    line=list(reader)[10]
    
    for i in range(3):
        filename = os.path.basename(line[i].strip())
        path = PATH + line[i].strip()
        _image = cv2.imread(path)
        cv2.imwrite("./examples/origin_"+filename,_image)
        image = preprocess_image(_image)
        print(filename)
        cv2.imwrite("./examples/preprocessd_"+filename,image)
        image,_ = argument(_image,0)
        cv2.imwrite("./examples/argument_"+filename,image)
    