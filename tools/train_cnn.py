import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.utils import np_utils


def main():
    input_dir = "../images"
    epochs = 30
    num_artist = len(
        [
            name
            for name in os.listdir(input_dir)
            if name != ".DS_Store" and name != ".gitkeep"
        ]
    )
    X_train, X_test, y_train, y_test = np.load("../npy/artists.npy", allow_pickle=True)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_artist)
    y_test = np_utils.to_categorical(y_test, num_artist)

    model = train(X_train, X_test, y_train, y_test, num_artist)
    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    history = model.fit(
        X_train, y_train, batch_size=32, epochs=epochs, validation_split=0.1
    )
    hdf5_file = f"../model/artist-model_{num_artist}_{epochs}.hdf5"
    model.save_weights(hdf5_file)
    json_string = model.to_json()
    with open("../model/cnn_model.json", mode="w") as f:
        f.write(json_string)
    plot_model(history, date_str)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss=", score[0])
    print("Test accuracy=", score[1])
    print(
        [
            name
            for name in os.listdir(input_dir)
            if name != ".DS_Store" and name != ".gitkeep"
        ]
    )


def train(X_train, X_test, y_train, y_test, num_artist):
    model = Sequential()
    # Keras official https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_artist))
    model.add(Activation("softmax"))

    model.summary()
    opt = RMSprop(lr=0.0001, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    return model


def plot_model(history, date_str):
    epochs = len(history.history["accuracy"])
    plt.figure()
    plt.plot(range(1, epochs + 1), history.history["accuracy"], "-o")
    plt.plot(range(1, epochs + 1), history.history["val_accuracy"], "-o")
    plt.title("model train accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.grid()
    plt.legend(["acc", "val_acc"], loc="best")
    plt_file = f"../result_model/learning_curve_acc_{date_str}.jpg"
    plt.savefig(plt_file)
    plt.close()

    plt.figure()
    plt.plot(range(1, epochs + 1), history.history["loss"], "-o")
    plt.plot(range(1, epochs + 1), history.history["val_loss"], "-o")
    plt.title("model train loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.grid()
    plt.legend(["loss", "val_loss"], loc="best")
    plt_file = f"../result_model/learning_curve_loss_{date_str}.jpg"
    plt.savefig(plt_file)
    plt.close()
    return


if __name__ == "__main__":
    main()
