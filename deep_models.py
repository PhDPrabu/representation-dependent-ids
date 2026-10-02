"""
Deep-learning model definitions for the representation-dependent IDS
reproducibility repository.

The implementations in this module are based on the available full
deep-learning experiment source used during the study. The documented
architectures are:

    1. 1D CNN
    2. Bidirectional LSTM (BiLSTM)

The source experiment uses tabular representation features reshaped to:

    (samples, features, 1)

Training configuration documented in the source:
    - Adam optimizer
    - learning rate = 0.001
    - categorical crossentropy
    - batch size = 128
    - maximum epochs = 50
    - early stopping on validation loss
    - patience = 8
    - restore_best_weights = True

CNN architecture:
    Conv1D(filters=64, kernel_size=3, activation='relu',
           padding='same')
    Dropout(0.3)
    GlobalMaxPooling1D
    Dense(number_of_classes, activation='softmax')

BiLSTM architecture:
    Bidirectional(LSTM(128, dropout=0.3, return_sequences=True))
    Bidirectional(LSTM(128, dropout=0.3, return_sequences=False))
    Dense(number_of_classes, activation='softmax')

The module does not perform data preprocessing. Training/validation/test
preprocessing is handled separately so that training-only fitting rules can
be enforced consistently.
"""

from __future__ import annotations

from typing import Optional

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    GlobalMaxPooling1D,
    Input,
    LSTM,
)
from tensorflow.keras.optimizers import Adam


MODEL_NAMES = (
    "CNN",
    "BiLSTM",
)


def build_cnn(
    n_features: int,
    n_classes: int = 3,
) -> Model:
    """
    Build the documented 1D CNN architecture.

    Parameters
    ----------
    n_features:
        Number of input representation features.
    n_classes:
        Number of target classes. The study uses three classes.

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled CNN model.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    inputs = Input(
        shape=(n_features, 1),
        name="input",
    )

    x = Conv1D(
        filters=64,
        kernel_size=3,
        activation="relu",
        padding="same",
        name="conv1d",
    )(inputs)

    x = Dropout(
        rate=0.3,
        name="dropout",
    )(x)

    x = GlobalMaxPooling1D(
        name="global_max_pooling1d",
    )(x)

    outputs = Dense(
        units=n_classes,
        activation="softmax",
        name="output",
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="CNN",
    )


def build_bilstm(
    n_features: int,
    n_classes: int = 3,
) -> Model:
    """
    Build the documented two-stage Bidirectional LSTM architecture.

    The first BiLSTM returns the complete sequence so that the second
    recurrent stage receives the sequence output. The second BiLSTM returns
    its final representation for the softmax classifier.
    """
    if n_features <= 0:
        raise ValueError("n_features must be positive.")

    if n_classes < 2:
        raise ValueError("n_classes must be at least 2.")

    inputs = Input(
        shape=(n_features, 1),
        name="input",
    )

    x = Bidirectional(
        LSTM(
            units=128,
            dropout=0.3,
            return_sequences=True,
        ),
        name="bidirectional_lstm_1",
    )(inputs)

    x = Bidirectional(
        LSTM(
            units=128,
            dropout=0.3,
            return_sequences=False,
        ),
        name="bidirectional_lstm_2",
    )(x)

    outputs = Dense(
        units=n_classes,
        activation="softmax",
        name="output",
    )(x)

    return Model(
        inputs=inputs,
        outputs=outputs,
        name="BiLSTM",
    )


def build_model(
    model_name: str,
    n_features: int,
    n_classes: int = 3,
) -> Model:
    """
    Build one of the supported deep-learning architectures.

    Parameters
    ----------
    model_name:
        ``CNN`` or ``BiLSTM``.
    n_features:
        Number of input representation features.
    n_classes:
        Number of output classes.
    """
    normalized_name = model_name.strip().upper()

    if normalized_name == "CNN":
        return build_cnn(
            n_features=n_features,
            n_classes=n_classes,
        )

    if normalized_name == "BILSTM":
        return build_bilstm(
            n_features=n_features,
            n_classes=n_classes,
        )

    raise ValueError(
        f"Unsupported deep-learning model '{model_name}'. "
        f"Supported models: {', '.join(MODEL_NAMES)}."
    )


def compile_model(
    model: Model,
    learning_rate: float = 0.001,
) -> Model:
    """
    Compile a model using the documented training configuration.

    Configuration:
        optimizer = Adam
        learning rate = 0.001
        loss = categorical_crossentropy
        metric = accuracy
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate,
        ),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def create_early_stopping(
    patience: int = 8,
) -> tf.keras.callbacks.EarlyStopping:
    """
    Create the documented early-stopping callback.

    Validation loss is monitored and the best weights are restored.
    """
    if patience < 0:
        raise ValueError("patience must be non-negative.")

    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
    )


def build_compiled_model(
    model_name: str,
    n_features: int,
    n_classes: int = 3,
    learning_rate: float = 0.001,
) -> Model:
    """Build and compile a supported deep-learning model."""
    model = build_model(
        model_name=model_name,
        n_features=n_features,
        n_classes=n_classes,
    )

    return compile_model(
        model=model,
        learning_rate=learning_rate,
    )


def model_configuration() -> dict:
    """
    Return the documented deep-learning configuration as a dictionary.

    This is useful for experiment logging and repository metadata.
    """
    return {
        "CNN": {
            "name": "1D CNN",
            "input_shape": "(n_features, 1)",
            "conv1d_filters": 64,
            "conv1d_kernel_size": 3,
            "conv1d_activation": "relu",
            "conv1d_padding": "same",
            "dropout": 0.3,
            "pooling": "GlobalMaxPooling1D",
            "output_activation": "softmax",
        },
        "BiLSTM": {
            "name": "Bidirectional LSTM",
            "input_shape": "(n_features, 1)",
            "first_lstm_units": 128,
            "first_lstm_dropout": 0.3,
            "first_lstm_return_sequences": True,
            "second_lstm_units": 128,
            "second_lstm_dropout": 0.3,
            "second_lstm_return_sequences": False,
            "output_activation": "softmax",
        },
        "training": {
            "optimizer": "Adam",
            "learning_rate": 0.001,
            "loss": "categorical_crossentropy",
            "metric": "accuracy",
            "batch_size": 128,
            "epochs": 50,
            "early_stopping_monitor": "val_loss",
            "early_stopping_patience": 8,
            "restore_best_weights": True,
        },
    }


def set_random_seed(seed: int) -> None:
    """
    Set TensorFlow's random seed for a reproducibility run.

    The experiment runner should also seed Python's ``random`` module and
    NumPy. This function handles the TensorFlow component.
    """
    tf.random.set_seed(seed)


def prepare_tabular_input(
    features,
) -> "tf.Tensor":
    """
    Convert a two-dimensional feature matrix to the model input layout.

    Expected source shape:
        (samples, features)

    Returned shape:
        (samples, features, 1)

    The conversion itself does not perform scaling, imputation, or any
    other data-dependent transformation.
    """
    tensor = tf.convert_to_tensor(features, dtype=tf.float32)

    if tensor.shape.rank != 2:
        raise ValueError(
            "Deep-learning tabular input must have shape "
            "(samples, features)."
        )

    return tf.expand_dims(tensor, axis=-1)


if __name__ == "__main__":
    print("Deep-learning model configuration self-check:")

    for model_name in MODEL_NAMES:
        model = build_model(
            model_name=model_name,
            n_features=10,
            n_classes=3,
        )

        print(f"\n{model_name}")
        model.summary()

    print("\nCompiling CNN self-check...")
    cnn = build_compiled_model(
        model_name="CNN",
        n_features=10,
        n_classes=3,
    )
    print("CNN compiled successfully.")

    print("\nTraining configuration:")
    configuration = model_configuration()["training"]
    for key, value in configuration.items():
        print(f"  {key}: {value}")
