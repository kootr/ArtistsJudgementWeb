import os
from PIL import Image
import numpy as np

from keras.models import model_from_json
from keras.optimizers import RMSprop


def main(target_image_path, hdf5_path):
    X = []
    image_size = 200
    image_dir = "./images"
    artistname = [name for name in os.listdir(image_dir) if name != ".DS_Store"]
    num_artist = len(artistname)
    img = Image.open(target_image_path)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    input_data = np.asarray(img)
    X.append(input_data)
    X = np.array(X)
    model = build_model(num_artist, image_size, hdf5_path)
    result_score = model.predict([X])[0]

    h_indexes = result_score.argsort()[::-1]

    return h_indexes, artistname, result_score


def build_model(num_artist, image_size, hdf5_path):
    opt = RMSprop(lr=0.0001, decay=1e-6)

    json_string = open("./model/cnn_model.json").read()
    model = model_from_json(json_string)

    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    model.load_weights(hdf5_path)

    return model


if __name__ == "__main__":
    main("./images/Claude_Monet/Claude_Monet_2.jpg", "artist-model_15_")
