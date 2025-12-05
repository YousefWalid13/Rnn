# cnn_mnist.py
"""
Simple CNN on MNIST.
Run:
    python cnn_mnist.py
Requires:
    pip install tensorflow numpy matplotlib
"""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
import os

def build_cnn(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def prepare_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # normalize and reshape
    x_train = x_train.astype('float32') / 255.0
    x_test  = x_test.astype('float32')  / 255.0
    x_train = np.expand_dims(x_train, -1)  # (N,28,28,1)
    x_test  = np.expand_dims(x_test, -1)
    return (x_train, y_train), (x_test, y_test)

def main():
    (x_train, y_train), (x_test, y_test) = prepare_data()
    model = build_cnn(input_shape=(28,28,1), num_classes=10)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    # train (use small epochs for quick runs)
    history = model.fit(x_train, y_train,
                        epochs=5,
                        batch_size=128,
                        validation_split=0.1)
    # evaluate
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc:.4f}")

    # save model
    out_dir = "cnn_mnist_model"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    model.save(os.path.join(out_dir, "mnist_cnn.h5"))
    print(f"Saved model to {out_dir}/mnist_cnn.h5")

    # show some predictions
    preds = model.predict(x_test[:9])
    preds_labels = np.argmax(preds, axis=1)
    plt.figure(figsize=(6,6))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(x_test[i].squeeze(), cmap='gray')
        plt.title(f"pred: {preds_labels[i]} (true: {y_test[i]})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
