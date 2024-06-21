![image](https://github.com/PestPatrol/pestpatrol-machine-learning/assets/89710992/95fabd66-ddf1-4148-b596-087c049fd056)

# PestPatrol Machine Learning
PestPatrol model using CNN method and Tensorflow framework

## About Project
PestPatrol is an Android application designed for early detection of rice plant diseases using advanced machine learning models. This project is built using Python, TensorFlow, and several other libraries, utilizing a Convolutional Neural Network (CNN) architecture based on the pre-trained DenseNet201 model.

## Table of Contents
- [Features](#features)
- [Libraries](#libraries)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Dataset](#dataset)
- [Image Preprocessing](#image-preprocessing)
- [Training](#training)
- [Evaluation](#evaluation)
- [Inference](#inference)
- [Model Conversion](#model-conversion)

## Features
- Detection of multiple rice leaf diseases using CNN.
- Data augmentation for robust model training.
- Visualization of training metrics and results.
- Conversion of trained model to TensorFlow.js for deployment

## Libraries
- TensorFlow
- TensorFlow.js
- NumPy
- Matplotlib
- Seaborn


## Project Structure
```sh
rice_leaf_disease_detection/
├── dataset/
│   └── riceleafs_extracted/
├── models/
│   └── model.json
└── notebooks/
    └── Rice_Leaf_Disease_Detection.ipynb
```

## Installation
Install Kaggle to download the dataset:
```sh
pip install kaggle
```
Install TensorFlow 2.15.0:
```sh
pip install TensorFlow==2.15.0
```
Install tensorflow-decision-forests 1.8.1:
```sh
pip install tensorflow-decision-forests==1.8.1
```

## Dataset
The dataset contains a collection of images of diseases in rice which are divided into 4 categories of types:
- BrownSpot
- Healthy
- Hispa
- Leaf blast

Link dataset : https://www.kaggle.com/datasets/shayanriyaz/riceleafs

## Image Preprocessing
```sh
import cv2
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 10))
for img, label in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(4, 6, i + 1)

        img_np = img[i].numpy().astype("uint8")
        blur_img = cv2.GaussianBlur(img_np, (5, 5), 0)
        hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2HSV)

        lower_green = (25, 40, 40)
        upper_green = (100, 255, 255)
        mask = cv2.inRange(hsv_img, lower_green, upper_green)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        green_objects = cv2.bitwise_and(img_np, img_np, mask=mask)
        green_objects_rgb = cv2.cvtColor(green_objects, cv2.COLOR_BGR2RGB)
        green_objects_rgb = cv2.convertScaleAbs(green_objects_rgb, alpha=1.5, beta=50)

        plt.imshow(green_objects_rgb)
        plt.title(class_names[label[i]])
        plt.axis("off")
plt.show()

```
## Training
```sh
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

input_tensor = Input(shape=(224, 224, 3))
base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor)
base_model.trainable = True

fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[reduce_lr])
```
## Evaluation
```sh
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
```
## Inference
```sh
def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\nPredicted: {predicted_class}, Confidence: {confidence}%")
        plt.axis("off")
plt.show()
```
## Model Conversion
Convert the trained model to TensorFlow.js format:
```sh
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model, 'models')
```
