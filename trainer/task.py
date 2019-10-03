"""Trains a Keras Model to Predict the Artist of a Painting"""

import argparse
import os
import datetime
import tensorflow as tf
import subprocess
import sys
from PIL import Image
sys.modules['Image'] = Image

from trainer import utils
from trainer import model as m


def get_args():
    """Argument parser

    Returns:
        A Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',
        type=str,
        required=True,
        help='local or GCS location for writing checkpoints and exporting models')
    parser.add_argument(
        '--train-file',
        type=str,
        default='gs://artwork_data_bucket/datasets/artworks.zip',
        help='the location of the dataset')
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=100,
        help='number of times to go through the data, default=20')
    parser.add_argument(
        '--batch-size',
        type=int,
        default=128,
        help="the number of records to read for each training step, default=128"
    )
    parser.add_argument(
        '--learning-rate',
        default=2e-5,
        type=float,
        help='learning rate for gradient descent, default=0.01')
    args, _ = parser.parse_known_args()
    hparams = tf.contrib.training.HParams(**args.__dict__)

    return hparams


def train_and_evaluate(hparams):
    """Trains and evaluates the keras model

    Uses the Keras model defined in model.py and trains on data loaded and
    preprocessed in preprocessing.py. Saves the trained model in TensorFlow
    SavedModel format to the path defined by the --job-dir argument.

    Args:
        args: dictionary of arguments - see get_args() for details
    """
    utils.load_data(hparams.train_file)

    model = m.create_keras_model(hparams)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        'train',
        target_size=(500, 500),
        batch_size=hparams.batch_size,
        class_mode='categorical')

    validation_generator = train_datagen.flow_from_directory(
        'validation',
        target_size=(500, 500),
        batch_size=hparams.batch_size,
        class_mode='categorical')

    cp = tf.keras.callbacks.ModelCheckpoint(filepath="artwork_cnn.h5",
                         save_best_only=True,
                         verbose=0)

    tb = tf.keras.callbacks.TensorBoard(log_dir='./logs',
                     histogram_freq=0,
                     write_graph=True,
                     write_images=True)

    es = tf.keras.callbacks.EarlyStopping(monitor='val_acc', mode='max', min_delta=1)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=hparams.num_epochs,
        validation_data=validation_generator,
        validation_steps=30,
        verbose=1,
        callbacks=[cp, tb, es])

    model_filename = 'final_artwork_cnn.h5'
    model.save(model_filename)
    model_folder = datetime.datetime.now().strftime('imdb_%Y%m%d_%H%M%S')

    gcs_model_path = os.path.join('gs://', hparams.bucket_name, 'results', model_folder, model_filename)
    subprocess.check_call(['gsutil', 'cp', model_filename, gcs_model_path], stderr=sys.stdout)
    os.remove(model_filename)

if __name__ == '__main__':
    hparams = get_args()
    train_and_evaluate(hparams)