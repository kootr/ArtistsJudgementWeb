import os
import datetime
import numpy as np
from keras.models import Sequential
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import matplotlib.pyplot as plt


def main():
    input_dir = "./images"
    num_classes = len(
        [name for name in os.listdir(input_dir) if name != "upload_images"]
    )
    X_train, X_test, y_train, y_test = np.load("./npy/artists.npy", allow_pickle=True)
    X_train = X_train.astype("float") / 255
    X_test = X_test.astype("float") / 255
    y_train = np_utils.to_categorical(y_train, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    model = train(X_train, X_test, y_train, y_test, num_classes)
    date_str = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    history = model.fit(X_train, y_train, batch_size=32, epochs=100)
    hdf5_file = f"./model/artist-model_{num_classes}.hdf5"
    model.save_weights(hdf5_file)
    plot_model(history, date_str)

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss=", score[0])
    print("Test accuracy=", score[1])


def train(X_train, X_test, y_train, y_test, num_classes):
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
    model.add(Dense(num_classes))
    model.add(Activation("softmax"))
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    return model


def plot_model(history, date_str):
    acc = history.history["accuracy"]
    loss = history.history["loss"]
    plt.figure()
    epochs = range(len(acc))
    plt.plot(epochs, acc, "bo", label="training acc")
    # plt.plot(epochs, val_acc, 'b' , label= 'validation acc')
    plt.plot(epochs, loss, "b", label="training loss")
    # plt.plot(epochs, val_loss, 'b' , label= 'validation loss')
    plt.title("Training accuracy and loss")
    plt.legend()
    plt_file = f"./learning_curve_{date_str}.jpg"
    plt.savefig(plt_file)
    return


if __name__ == "__main__":
    main()
