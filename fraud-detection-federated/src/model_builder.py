"""
Builds configurable Keras models for binary classification.
Supports different architectures and training configurations.
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf

def build_model(input_shape=1, 
                architecture='simple',
                optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'],
                dropout_rate=0.2,
                show_summary=True):
    """
    Builds and compiles a Keras Sequential model.

    Args:
        input_shape (int): Number of input features.
        architecture (str): One of ['simple', 'wide', 'deep'].
        optimizer (str or tf.keras.optimizers.Optimizer): Optimizer to use.
        loss (str): Loss function.
        metrics (list): List of metrics to evaluate.
        dropout_rate (float): Dropout rate if applicable.
        show_summary (bool): If True, prints the model summary.

    Returns:
        Compiled Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    if architecture == 'simple':
        model.add(Dense(16, activation='relu'))
    elif architecture == 'wide':
        model.add(Dense(64, activation='relu'))
        model.add(Dense(64, activation='relu'))
    elif architecture == 'deep':
        model.add(Dense(64, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(dropout_rate))
        model.add(Dense(16, activation='relu'))
    else:
        raise ValueError("Invalid architecture. Choose from 'simple', 'wide', or 'deep'.")

    model.add(Dense(1, activation='sigmoid'))

    # Optimizer selection
    if isinstance(optimizer, str):
        if optimizer == 'adam':
            optimizer = Adam()
        elif optimizer == 'sgd':
            optimizer = SGD()
        elif optimizer == 'rmsprop':
            optimizer = RMSprop()
        else:
            raise ValueError("Unsupported optimizer name.")

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    if show_summary:
        model.summary()

    return model

# Example usage
if __name__ == "__main__":
    # Dummy data to test compilation
    import numpy as np

    X_dummy = np.random.rand(100, 5)
    y_dummy = np.random.randint(0, 2, 100)

    print("Building and training 'deep' model with 5 inputs...")
    model = build_model(input_shape=5, architecture='deep', optimizer='ada
