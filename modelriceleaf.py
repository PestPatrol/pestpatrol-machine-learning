# -*- coding: utf-8 -*-
"""modelRiceLeaf

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1y9gyLd_9hxrxSbdj-XrC4Gu2hMeltZiC

## Import Kaggle dan Dataset
"""

#install kaggle

!pip install -q kaggle

from google.colab import files
files.upload()

!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download shayanriyaz/riceleafs

import zipfile

# Path ke file zip
zip_path = "/content/riceleafs.zip"

# Membuka file zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # Mendapatkan daftar nama file di dalam zip
    file_list = zip_ref.namelist()

    # Misalnya, untuk mengekstrak semua file dalam zip
    zip_ref.extractall("/content/riceleafs_extracted")

# Menampilkan daftar file yang diekstrak
print("File yang diekstrak:")
print(file_list)

"""## Import Package"""

!pip install tensorflowjs

import tensorflowjs as tfjs
print(tfjs.__version__)

!pip install TensorFlow==2.15.0
!pip install tensorflow-decision-forests==1.8.1

import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tensorflow.keras import models, layers
import matplotlib.pyplot as plt
from IPython.display import HTML
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG19
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import DenseNet201
from keras.layers import GlobalAveragePooling2D, Dropout, Dense
from keras.models import Model
from tensorflow.keras.layers import Input

"""## Load Data"""

BATCH_SIZE = 32
IMAGE_SIZE = 224
CHANNELS=3

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "/content/riceleafs_extracted/RiceLeafs/train",
    seed=123,
    shuffle=True,
    image_size=(IMAGE_SIZE,IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names
class_names

"""## Image Preprocessing"""

plt.figure(figsize=(15, 10))
for img, label in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(4, 6, i + 1)

        # Preprocess the image
        img_np = img[i].numpy().astype("uint8")
        blur_img = cv2.GaussianBlur(img_np, (5, 5), 0)
        hsv_img = cv2.cvtColor(blur_img, cv2.COLOR_RGB2HSV)

        # Define the range for green color
        lower_green = (25, 40, 40)
        upper_green = (100, 255, 255)

        # Create mask for green color
        mask = cv2.inRange(hsv_img, lower_green, upper_green)

        # Apply morphological operations for better mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Apply the mask to extract the green objects
        green_objects = cv2.bitwise_and(img_np, img_np, mask=mask)

        # Convert to RGB format
        green_objects_rgb = cv2.cvtColor(green_objects, cv2.COLOR_BGR2RGB)

        # Increase brightness and contrast
        green_objects_rgb = cv2.convertScaleAbs(green_objects_rgb, alpha=1.5, beta=50)

        # Display images
        plt.imshow(green_objects_rgb)
        plt.title(class_names[label[i]])
        plt.axis("off")
plt.show()

# Displaying sample images after preprocessing
plt.figure(figsize=(10, 10))
for images, labels in dataset.take(1):
    for i in range(12):
        ax = plt.subplot(3, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")

"""## Split Data"""

# Function to partition the dataset
def get_dataset_partitions_tf(ds, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    ds_size = len(ds)

    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

# Partition the dataset
train_ds, val_ds, test_ds = get_dataset_partitions_tf(dataset)

"""Data Augmentation"""

# Data augmentation function
def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
    image = tf.image.random_saturation(image, lower=0.9, upper=1.1)
    image = tf.image.random_hue(image, max_delta=0.1)
    return image, label

# Apply data augmentation to the training dataset
train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Display some augmented images
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")
plt.show()

"""MODEL"""

# Load the pre-trained DenseNet201 model without the top layer
input_tensor = Input(shape=(224, 224, 3))
base_model = DenseNet201(weights='imagenet', include_top=False, input_tensor=input_tensor)

base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the 'fine_tune_at' layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Create the final model
model = Model(inputs=input_tensor, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[reduce_lr]
)

# Clear the session
tf.keras.backend.clear_session()

scores = model.evaluate(test_ds)

scores

history

history.params

history.history.keys()

type(history.history['loss'])

len(history.history['loss'])

history.history['loss'][:5] # show loss for first 5 epochs

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Assuming you have the history object from the model.fit call
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Get the number of epochs actually completed
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

"""Run prediction on a sample image"""

import numpy as np
for images_batch, labels_batch in test_ds.take(1):

    first_image = images_batch[0].numpy().astype('uint8')
    first_label = labels_batch[0].numpy()

    print("first image to predict")
    plt.imshow(first_image)
    print("actual label:",class_names[first_label])

    batch_prediction = model.predict(images_batch)
    print("predicted label:",class_names[np.argmax(batch_prediction[0])])

"""Write a function for inference"""

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(images[i].numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

"""Now run inference on few sample images"""

import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(15, 15))
for images, labels in test_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))

        predicted_class, confidence = predict(model, images[i].numpy())
        actual_class = class_names[labels[i]]

        plt.title(f"Actual: {actual_class},\n Predicted: {predicted_class}.\n Confidence: {confidence}%")

        plt.axis("off")

#heatmap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Prediksi target dengan model
predictions = model.predict(test_ds)

# Konversi prediksi menjadi label kelas
predicted_labels = np.argmax(predictions, axis=1)

# Ambil label sebenarnya dari data uji
true_labels = np.concatenate([y for x, y in test_ds], axis=0)

# Hitung matriks kebingungan
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Visualisasikan heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

from sklearn.metrics import classification_report

# Buat laporan klasifikasi
class_report = classification_report(true_labels, predicted_labels)

print("Classification Report:\n", class_report)

"""convert model"""

tfjs.converters.save_keras_model(model, 'models')

!ls

!cat model.json

from google.colab import files

files.download('models')

import tensorflowjs as tfjs
tfjs.converters.save_keras_model(model, 'models')

import shutil
# Lokasi folder yang ingin Anda kompres
folder_path = '/content/models'

# Lokasi tempat menyimpan file ZIP
zip_path = '/content/models.zip'

# Kompres folder menjadi file ZIP
shutil.make_archive('/content/models', 'zip', folder_path)

# Lokasi file ZIP yang ingin diunggah
zip_file = '/content/models.zip'

"""STOP"""

import os

# Tentukan direktori untuk menyimpan model
model_directory = "../models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i) for i in os.listdir(model_directory) if os.path.isfile(os.path.join(model_directory, i))]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan model dengan versi terbaru
model.save(f"{model_directory}/{model_version}")

# Simpan model
model.save("densenet4.h5")

# Simpan arsitektur model ke dalam format JSON
model_json = model.to_json()
with open(f"{model_directory}/{model_version}/model.json", "w") as json_file:
    json_file.write(model_json)

# Simpan bobot model ke dalam format HDF5
model.save_weights(f"{model_directory}/{model_version}/model_weights.h5")

print(f"Model berhasil disimpan dengan versi: {model_version}")

import os
import json
from tensorflow.keras.models import model_from_json

# Tentukan direktori untuk menyimpan model
model_directory = "../models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i) for i in os.listdir(model_directory) if i.isdigit()]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan model dengan versi terbaru dalam format .h5
model.save(f"{model_directory}/{model_version}/model.h5")

# Simpan arsitektur model ke dalam format JSON
model_json = model.to_json()
with open(f"{model_directory}/{model_version}/model.json", "w") as json_file:
    json_file.write(model_json)

# Simpan bobot model ke dalam format HDF5
model.save_weights(f"{model_directory}/{model_version}/model_weights.h5")

print(f"Model berhasil disimpan dengan versi: {model_version}")

# Baca arsitektur model dari file JSON
with open(f"{model_directory}/{model_version}/model.json", "r") as json_file:
    model_json = json_file.read()

# Buat model dari arsitektur JSON
model = model_from_json(model_json)

# Muat bobot ke model
model.load_weights(f"{model_directory}/{model_version}/model_weights.h5")

print("Model berhasil dimuat dari file JSON dan bobot.")

# Simpan model
model.save("densenet4.json")

import os
import json
from tensorflow.keras.models import model_from_json

# Tentukan direktori untuk menyimpan model
model_directory = "../models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i.split('.')[0]) for i in os.listdir(model_directory) if i.endswith('.json')]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan arsitektur model dalam format JSON
model_json = model.to_json()
with open(f"{model_directory}/{model_version}.json", "w") as json_file:
    json_file.write(model_json)

# Simpan bobot model
model.save_weights(f"{model_directory}/{model_version}_weights.h5")

# Contoh untuk memuat kembali model dari file JSON dan bobotnya
with open(f"{model_directory}/{model_version}.json", "r") as json_file:
    loaded_model_json = json_file.read()

loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(f"{model_directory}/{model_version}_weights.h5")

# Pastikan untuk mengompilasi ulang model jika diperlukan
# loaded_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

pip install pydrive

import os
import json
from tensorflow.keras.models import model_from_json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Tentukan direktori untuk menyimpan model
model_directory = "../models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i.split('.')[0]) for i in os.listdir(model_directory) if i.endswith('.json')]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan arsitektur model dalam format JSON
model_json = model.to_json()
json_path = f"{model_directory}/{model_version}.json"
with open(json_path, "w") as json_file:
    json_file.write(model_json)

# Simpan bobot model
weights_path = f"{model_directory}/{model_version}_weights.h5"
model.save_weights(weights_path)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials
# Autentikasi dan buat koneksi PyDrive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

# Unggah file JSON
json_file = drive.CreateFile({'title': f'{model_version}.json'})
json_file.SetContentFile(json_path)
json_file.Upload()

# Unggah file H5
weights_file = drive.CreateFile({'title': f'{model_version}_weights.h5'})
weights_file.SetContentFile(weights_path)
weights_file.Upload()

print("Model and weights are successfully uploaded to Google Drive.")

import os
from tensorflow.keras.models import load_model
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Tentukan direktori untuk menyimpan model
model_directory = "../models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i.split('.')[0]) for i in os.listdir(model_directory) if i.endswith('.h5')]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan model dengan versi terbaru
model_path = f"{model_directory}/{model_version}.h5"
model.save(model_path)

# Autentikasi dan inisialisasi Google Drive
gauth = GoogleAuth()
gauth.LocalWebserverAuth()  # Ini akan membuka browser untuk autentikasi
drive = GoogleDrive(gauth)

# Unggah file .h5 ke Google Drive
h5_file = drive.CreateFile({'title': f'{model_version}.h5'})
h5_file.SetContentFile(model_path)
h5_file.Upload()

print("Model .h5 successfully uploaded to Google Drive.")

import os
import json
import numpy as np
from tensorflow.keras.models import model_from_json
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive

# Tentukan direktori untuk menyimpan model
model_directory = "./models"

# Buat direktori jika belum ada
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Cari versi model terbaru
model_files = [int(i.split('.')[0]) for i in os.listdir(model_directory) if i.endswith('.json')]
if model_files:
    model_version = max(model_files) + 1
else:
    model_version = 1

# Simpan arsitektur model dalam format JSON
model_json = model.to_json()
json_path = f"{model_directory}/{model_version}.json"
with open(json_path, "w") as json_file:
    json_file.write(model_json)

# Fungsi untuk menyimpan bobot model dalam format sharded .bin
def save_model_weights_in_shards(model, directory, model_version, shard_size=1024 * 1024 * 10):
    weights = model.get_weights()
    flat_weights = np.concatenate([w.flatten() for w in weights])
    total_size = flat_weights.nbytes
    shard_count = (total_size // shard_size) + 1

    shard_paths = []
    for shard_idx in range(shard_count):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, total_size)
        shard = flat_weights[start_idx:end_idx]
        shard_path = os.path.join(directory, f"{model_version}_weights_{shard_idx}.bin")
        shard.tofile(shard_path)
        shard_paths.append(shard_path)

    return shard_paths

# Simpan bobot model dalam format sharded .bin
shard_paths = save_model_weights_in_shards(model, model_directory, model_version)

from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Autentikasi dan buat koneksi PyDrive
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)


# Unggah file JSON ke Google Drive
json_file = drive.CreateFile({'title': f'{model_version}.json'})
json_file.SetContentFile(json_path)
json_file.Upload()

# Unggah file .bin ke Google Drive
for shard_path in shard_paths:
    shard_file = drive.CreateFile({'title': os.path.basename(shard_path)})
    shard_file.SetContentFile(shard_path)
    shard_file.Upload()

print("Model JSON and sharded .bin files successfully uploaded to Google Drive.")

