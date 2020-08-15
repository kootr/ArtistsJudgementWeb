import os
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense


def main(target_image_path, hdf5_path):
    X = []
    image_size = 50
    image_dir = "./images"
    artistname = [
        name
        for name in os.listdir(image_dir)
        if (name != "upload_images") and (name != ".DS_Store")
    ]
    num_artist = len(artistname)
    img = Image.open(target_image_path)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    input_data = np.asarray(img)
    X.append(input_data)
    X = np.array(X)
    model = build_model(num_artist, hdf5_path)
    result = model.predict([X])[0]

    predict = model.predict(X)
    for name, score in zip(artistname, result):
        print(f"{name}: {score*100}")
    # predictの中で一番大きい値のラベルを返す
    y = predict.argmax()
    return (artistname[y], target_image_path)


def build_model(num_artist, hdf5_path):
    model = Sequential()
    # Keras official https://github.com/keras-team/keras/blob/master/examples/cifar10_cnn.py
    model.add(Conv2D(32, (3, 3), padding="same", input_shape=(50, 50, 3)))
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
    model.compile(
        loss="categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"]
    )

    model.load_weights(hdf5_path)

    return model


if __name__ == "__main__":
    main("./images/Claude_Monet/Claude_Monet_2.jpg", "./model/artist-model_master.hdf5")
