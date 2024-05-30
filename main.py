import cv2
import numpy
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models

(training_images, training_labels), (testing_images, testing_lables) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255, testing_images / 255

class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', "Frog", 'Horse', 'Ship', 'Truck']

training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_lables = testing_lables[:4000]

model = models.load_model('image_classifier.model')

img = cv2.imread('horse.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img]) / 255)

index = np.argmax(prediction)
print(f'Prediction is {class_names[index]}')