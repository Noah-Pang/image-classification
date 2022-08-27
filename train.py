import numpy as np
import PIL
import tensorflow as tf
import pathlib

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from utils import load_config
from utils import load_dataset

config = load_config("./config.yaml")

def train(model_name: str, data_dir):
    train_ds, val_ds = load_dataset(data_dir)

    class_names = train_ds.class_names
    print(class_names)

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    data_augmentation = keras.Sequential(
      [
        layers.RandomFlip("horizontal",
                          input_shape=(config["img_height"],
                                      config["img_width"],
                                      3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
      ]
    )

    num_classes = len(class_names)

    model = Sequential([
      data_augmentation,
      layers.Rescaling(1./255, input_shape=(config["img_height"], config["img_width"], 3)),
      layers.Conv2D(16, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(32, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Conv2D(64, 3, padding='same', activation='relu'),
      layers.MaxPooling2D(),
      layers.Dropout(config["dropout_rate"]),
      layers.Flatten(),
      layers.Dense(128, activation='relu'),
      layers.Dense(num_classes)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=config["epochs"]
    )

    model.save("./models/" + model_name)

    return history
