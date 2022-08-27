import yaml
import tensorflow as tf

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

# def resize_image(img_path):
#     config = load_config("./config.yaml")
#
#     image = tf.keras.utils.load_img(
#         img_path, target_size=(config["img_height"], config["img_width"])
#     )
#     image = tf.image.convert_image_dtype(image, tf.float32)
#     image = tf.image.resize(image, (224, 224))
#     return image