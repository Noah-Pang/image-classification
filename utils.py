import yaml
import tensorflow as tf
import matplotlib.pyplot as plt

def load_config(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)

    return config

def load_dataset(data_dir):
    config = load_config("./config.yaml")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config["valid_split"],
        subset="training",
        seed=config["seed"],
        image_size=(config["img_height"], config["img_width"]),
        batch_size=config["batch_size"])

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=config["valid_split"],
        subset="validation",
        seed=config["seed"],
        image_size=(config["img_height"], config["img_width"]),
        batch_size=config["batch_size"])

    return train_ds, val_ds

def plot_accuracy(history, model_name):
    config = load_config("./config.yaml")
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(config["epochs"])

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
    plt.savefig("./models/" + model_name + "/accuracy_plot")

# def resize_image(img_path):
#     config = load_config("./config.yaml")
#
#     image = tf.keras.utils.load_img(
#         img_path, target_size=(config["img_height"], config["img_width"])
#     )
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, (224, 224))
#     return image