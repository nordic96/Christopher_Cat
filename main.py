from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model.neural_network_model import NeuralNetworkModel
from trainer.model_trainer import ModelTrainer, visualise_history

parser = argparse.ArgumentParser()
parser.add_argument('--mode', help='execution mode. train, test')
args = parser.parse_args()

_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True)
PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')
BATCH_SIZE = 20
EPOCHS = 10
IMG_HEIGHT = 150
IMG_WIDTH = 150
MODEL_FILENAME = 'christopher_model.h5'


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def load_val_train_data():
    train_dir = os.path.join(PATH, 'train')
    validation_dir = os.path.join(PATH, 'validation')

    train_cats_dir = os.path.join(train_dir, 'cats')  # directory with our training cat pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')  # directory with our training dog pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')  # directory with our validation cat pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')  # directory with our validation dog pictures

    num_cats_tr = len(os.listdir(train_cats_dir))
    num_dogs_tr = len(os.listdir(train_dogs_dir))

    num_cats_val = len(os.listdir(validation_cats_dir))
    num_dogs_val = len(os.listdir(validation_dogs_dir))

    total_train = num_cats_tr + num_dogs_tr
    total_val = num_cats_val + num_dogs_val

    print('total training cat images:', num_cats_tr)
    print('total training dog images:', num_dogs_tr)

    print('total validation cat images:', num_cats_val)
    print('total validation dog images:', num_dogs_val)
    print("--")
    print("Total training images:", total_train)
    print("Total validation images:", total_val)
    return train_dir, validation_dir


def generate_generator():
    train_image_generator = ImageDataGenerator(rescale=1. / 255,
                                               rotation_range=45,
                                               width_shift_range=.15,
                                               height_shift_range=.15,
                                               horizontal_flip=True,
                                               zoom_range=0.5)
    validation_image_generator = ImageDataGenerator(rescale=1. / 255)
    train_dir, validation_dir = load_val_train_data()

    train_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                          directory=train_dir,
                                                          shuffle=True,
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                          class_mode='binary')

    val_gen = validation_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
                                                             directory=validation_dir,
                                                             shuffle=True,
                                                             target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                             class_mode='binary')

    return train_gen, val_gen


def load(filename):
    img = image.load_img(filename, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img

# Total number of steps per epoch = total images (2000) = steps * batch size (100 * 20)
# Total number of validation steps = total images (1000) steps * batch size (50 * 20)
if __name__ == '__main__':
    if args.mode == 'train':
        neural_network = NeuralNetworkModel(IMG_HEIGHT, IMG_WIDTH, channel_size=3)
        train_data_gen, val_data_gen = generate_generator()
        model_trainer = ModelTrainer(train_data_gen=train_data_gen,
                                     steps_per_epoch=100,
                                     epochs=EPOCHS,
                                     validation_gen=val_data_gen,
                                     validation_steps=50,
                                     model=neural_network.model)
        history = model_trainer.train_model()
        visualise_history(history)

    if args.mode == 'predict':
        loaded_model = tf.keras.models.load_model('christopher_model.hdf5')
        loaded_model.summary()
        img_to_predict = load('data/dog.jpg')
        result = loaded_model.predict(img_to_predict, batch_size=1)
        print(result)
