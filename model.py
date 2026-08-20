from dotenv import load_dotenv
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.title("Bikes Vs. Cars CNN Model")

load_dotenv()

dataset_url = os.getenv("DATASET_URL")

train_dataset = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(180,180),
  batch_size=16)

val_dataset = tf.keras.utils.image_dataset_from_directory(
  dataset_url,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(180,180),
  batch_size=32)

AUTOTUNE = tf.data.AUTOTUNE

train_dataset_nm = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset_nm = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)


normalization_layer = layers.Rescaling(1./255)

normalized_train_dataset_nm = train_dataset_nm.map(lambda x, y: (normalization_layer(x), y))
normalized_val_dataset_nm = val_dataset_nm.map(lambda x, y: (normalization_layer(x), y))

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import F1Score, Precision, Recall

data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal", input_shape=(180, 180, 3)),
  layers.RandomZoom(0.1),
])

model = Sequential([
  data_augmentation,
  layers.Rescaling(1./255),
  layers.Conv2D(16,3,padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32,3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64,3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.2),
  layers.Flatten(),
  layers.Dropout(0.2),
  layers.Dense(128, activation='relu'),
  layers.Dense(len(train_dataset.class_names))
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

epochs = 10

history = model.fit(
  normalized_train_dataset_nm,
  validation_data=normalized_val_dataset_nm,
  epochs=epochs,
  callbacks=[callback]
)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

print(f"Train accuracy: {accuracy[-1]:.2f}")
print(f"Validation Accuracy: {val_accuracy[-1]:.2f}")

sample_dataset = train_dataset.take(2)

class_names = train_dataset.class_names

for images, labels in sample_dataset:
  for image in images:
    image_array = tf.keras.utils.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = model.predict(image_array)
    score = tf.nn.softmax(predictions[0])

    image = tf.clip_by_value(image, 0, 255)
    image = tf.cast(image, tf.uint8)

    plt.imshow(image)
    plt.title(f"{class_names[np.argmax(score)]} {100 * np.max(score)}")

    plt.show()
    

