import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model = load_model('BrainTumor.h5')

image = cv2.imread('g:\\BrainCancer\\uploads\\pred7.jpg')  # Replace with your image path
img = Image.fromarray(image)
img = img.resize((64, 64))
img = np.array(img)

input_img = np.expand_dims(img, axis=0)

print("mainTest.py - Image shape after resizing:", img.shape)
print("mainTest.py - Processed image:", img)

result = model.predict(input_img)
print("Predicted probabilities:", result)

threshold = 0.5
result_classes = (result > threshold).astype(int)
print("Predicted classes:", result_classes)
