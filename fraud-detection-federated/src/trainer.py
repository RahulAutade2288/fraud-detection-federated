"""
Trains a Keras model on local client data.
Supports early stopping, validation, and history visualization.
"""

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, X, y,
                epochs=10,
                batch_size=8,
                validation_split=0.2,
                patience=3,
                save_path=None,
                verbose=1,
                plot_history=True):
    """
    Trains the model on local data.

    Args:
        model: Compiled Keras model.
        X (np.ndarray): Features.
        y (np.ndarray): Labels.
        epochs (int): Number of epochs.
        batch_size (int): Batch size.
        validation_split (float): Portion of training data used for validation.
        patience (int): Early stopping patience.
        save_path (str): Optional path to save best model.
        verbose (int): Verbosity level.
        plot_history (bool): Plot training/validation loss.

    Returns:
        model: Trained model.
        history: Training history object.
    """
    callbacks = []
    if patience > 0:
        callbacks.append(EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True))
    if save_path:
        callbacks.append(ModelCheckpoint(filepath=save_path, monitor='val_loss', save_best_only=True))

    history = model.fit(
        X, y,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        callbacks=callbacks
    )

    if plot_history:
        plot_training_history(history)

    return model, history

def plot_training_history(history):
    """
    Plots training and validation accuracy/loss curves.
    """
    if history is None:
        return

    history_dict = history.history

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    if 'accuracy' in history_dict:
        plt.plot(history_dict['accuracy'], label='Train Acc')
        if 'val_accuracy' in history_dict:
            plt.plot(history_dict['val_accuracy'], label='Val Acc')
        plt.title("Accuracy over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history_dict['loss'], label='Train Loss')
    if 'val_loss' in history_dict:
        plt.plot(history_dict['val_loss'], label='Val Loss')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Test mode
if __name__ == "__main__":
    import numpy as np
    from model_builder import build_model  # Ensure this file exists or replace with inline model

    # Generate dummy data
    X_dummy = np.random.rand(100, 3)
    y_dummy = np.random.randint(0, 2, 100)

    model = build_model(input_shape=3, architecture='simple')
    model, history = train_model(model, X_dummy, y_dummy, epochs=5)
