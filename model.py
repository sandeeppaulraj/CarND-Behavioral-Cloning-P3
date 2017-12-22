# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 11:29:46 2017

@author: sandeep
"""

#importing some useful packages
import numpy as np
import cv2
import csv
import tensorflow as tf
import sklearn

from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.preprocessing import image
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Lambda
from keras.layers import Cropping2D
from keras.models import Sequential

lines = []

with open('../data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

# This is required since the first line is comprised of headers.
lines = lines[1:]

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

print(len(train_samples))
print(len(validation_samples))

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = '../data/IMG/'+batch_sample[0].split('/')[-1]
                image = cv2.imread(name)
                center_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

                image_flipped = np.copy(np.fliplr(image))
                image_flipped_rgb = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2RGB)
                images.append(image_flipped_rgb)
                angle_flipped = -center_angle
                angles.append(angle_flipped)
                
                name = '../data/IMG/'+batch_sample[1].split('/')[-1]
                image = cv2.imread(name)
                left_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                left_angle = center_angle + 0.085
                images.append(left_image)
                angles.append(left_angle)
                
                name = '../data/IMG/'+batch_sample[2].split('/')[-1]
                image = cv2.imread(name)
                right_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                right_angle = center_angle - 0.085
                images.append(right_image)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Conv2D(24, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Conv2D(36, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Conv2D(48, 5, 5, subsample=(2,2), activation = 'relu'))
model.add(Conv2D(64, 3, 3, subsample=(1,1), activation = 'relu'))
model.add(Conv2D(64, 3, 3, subsample=(1,1), activation = 'relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.4))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')

epochs = 2

history_object = model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=epochs)

model.save('model1.h5')