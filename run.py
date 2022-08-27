import tensorflow as tf
import numpy as np

from tensorflow import keras

from utils import load_config
from utils import load_dataset

def load_model(model_path, data_dir, images_path, images):
    model = keras.models.load_model(model_path)
    config = load_config("./config.yaml")
    train_ds, val_ds = load_dataset(data_dir)

    class_names = train_ds.class_names

    for i, image in enumerate(images):
        img = tf.keras.utils.load_img(
            images_path + image, target_size=(config["img_height"], config["img_width"])
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(
            "Image {} most likely belongs to {} with a {:.2f} percent confidence."
            .format(images[i], class_names[np.argmax(score)], 100 * np.max(score))
        )