"""Defines a Keras model and input function for training"""

import tensorflow as tf

def load_base_model(hparams):
    """Loads a Pretrained VGG16 Model for Transfer Learning

        Args:
            none
        Returns:
            A pretrained VGG16 Model
    """
    print("Loading base Model...")
    conv_base = tf.keras.applications.VGG16(weights="imagenet", include_top = False,
                                            input_shape=(500, 500, 3))
    print("Loaded")
    return conv_base

def create_keras_model(hparams):
    """Creates a Keras Model for Multi-class Classification.

    Args:
        input_dim: The number of input features
        learning_rate: Learning rate for training
        num_classes: The number of output classes

    Returns:
        the compiled Keras model (still needs to be trained)
    """

    conv_base = load_base_model(hparams)
    conv_base.trainable = False
    Flatten = tf.keras.layers.Flatten
    Dropout = tf.keras.layers.Dropout
    Dense = tf.keras.layers.Dense

    model = tf.keras.Sequential(
        [
            conv_base,
            Flatten(),
            Dropout(rate=0.5),
            Dense(512, activation='relu'),
            Dense(12, activation='softmax')
        ]
    )

    optimizer = tf.keras.optimizers.RMSprop(lr=hparams.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model