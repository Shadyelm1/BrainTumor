import cv2
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import normalize
from keras.layers import Activation


# Define your image directory
image_directory = 'datasets/'

# Initialize lists to store images and labels
dataset = []
label = []

input_size = 64

# Load images from 'no' folder
no_tumor_images = os.listdir(image_directory + 'no/')
for i, image_name in enumerate(no_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'no/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(0)

# Load images from 'yes' folder
yes_tumor_images = os.listdir(image_directory + 'yes/')
for i, image_name in enumerate(yes_tumor_images):
    if image_name.split('.')[1] == 'jpg':
        image = cv2.imread(image_directory + 'yes/' + image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((input_size, input_size))
        dataset.append(np.array(image))
        label.append(1)

# Convert lists to numpy arrays
dataset = np.array(dataset)
label = np.array(label)

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

# Normalize the data
x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

# Model Building
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(input_size, input_size, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x_train, y_train, 
          batch_size=16, 
          verbose=1, 
          epochs=10,
          validation_data=(x_test, y_test),
          shuffle=False)

model.save('BrainTumor.h5')
