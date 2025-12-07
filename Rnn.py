
"""
RNN (LSTM) for MNIST classification.
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os


def get_data():
    """Load MNIST dataset and normalize."""
    (train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()

    train_x = train_x.astype("float32") / 255.0
    test_x = test_x.astype("float32") / 255.0

    # RNN expects (batch, timesteps, features)
    return (train_x, train_y), (test_x, test_y)


def rnn_model(input_shape=(28, 28), classes=10):
    """Create a small LSTM model."""
    net = models.Sequential()
    
    # Slightly different architecture
    net.add(layers.LSTM(80, return_sequences=False, input_shape=input_shape))
    net.add(layers.Dropout(0.25))
    net.add(layers.Dense(100, activation="relu"))
    net.add(layers.Dense(classes, activation="softmax"))
    
    return net


def run():
    (train_x, train_y), (test_x, test_y) = get_data()

    model = rnn_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print(model.summary())

    history = model.fit(
        train_x,
        train_y,
        batch_size=128,
        epochs=5,
        validation_split=0.1
    )

    loss, acc = model.evaluate(test_x, test_y, verbose=2)
    print(f"\nTest Accuracy = {acc:.4f}")

    # Save model folder
    out_dir = "rnn_mnist_v2_model"
    os.makedirs(out_dir, exist_ok=True)
    model.save(os.path.join(out_dir, "rnn_model_v2.h5"))
    print("Model saved successfully.")

    # Show sample predictions
    preds = model.predict(test_x[:9])
    pred_labels = np.argmax(preds, axis=1)

    plt.figure(figsize=(6, 6))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.imshow(test_x[i], cmap="gray")
        plt.title(f"Pred: {pred_labels[i]} | True: {test_x[i]}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()