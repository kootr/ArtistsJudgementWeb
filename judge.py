import train as train
import os
from PIL import Image
import numpy as np


def evaluation(target_image_path, hdf5_path):
    image_size = 50
    input_dir = "data"
    artistname = [name for name in os.listdir(input_dir) if name != ".DS_Store"]
    X = []
    img = Image.open(target_image_path)
    img = img.convert("RGB")
    img = img.resize((image_size, image_size))
    in_data = np.asarray(img)
    X.append(in_data)
    X = np.array(X)

    model = train.TrainModel().train(X.shape[1:])
    model.load_weights("./model/artist-model.hdf5")
    predict = model.predict(X)

    # predictの中で一番大きい値のラベルを返す
    y = predict.argmax()
    return (artistname[y], target_image_path)


if __name__ == "__main__":
    evaluation("./testdata/img_1.jpg", "./model/artist-model.hdf5")
