import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import os

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from train import train
from run import load_model


dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)




history = train("dog", "./datasets/dog")

test_files = []
for (dir_path, dir_names, file_names) in os.walk("./testing/dog"):
    for i in range(len(file_names)):
        file_names[i] = dir_path + "/" + file_names[i]
    test_files.extend(file_names)
load_model("models/dog", "./datasets/dog", test_files)