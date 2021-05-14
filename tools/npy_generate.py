import glob
import math
import os
import random

import numpy as np
from PIL import Image

image_size = 200
num_data = 300
Image.LOAD_TRUNCATED_IMAGES = True


def main():
    train_data = []
    artistname = []
    image_dir = "../images"
    artistname = [
        name
        for name in os.listdir(image_dir)
        if name != ".DS_Store" and name != ".gitkeep"
    ]
    for name_idx, name in enumerate(artistname):
        train_cnt = 0
        try:
            print("---", name)
            artist_image_dir = image_dir + "/" + name
            files = glob.glob(artist_image_dir + "/*.jpg")
            for file_idx, file in enumerate(files):
                if train_cnt >= num_data:
                    break
                img = Image.open(file)
                img = img.convert("RGB")
                img = img.resize((image_size, image_size))

                # added for no augumentation
                # rotation
                for angle in range(-90, 1, 90):
                    img_r = img.rotate(angle)
                    img_np = np.asarray(img_r)
                    train_data.append([img_np, name_idx])
                    train_cnt += 1
                    # inverse
                    # img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                    # img_np = np.asarray(img_trans)
                    # train_data.append([img_np, name_idx])
                    # train_cnt += 1
                    if train_cnt >= num_data:
                        break
        except:
            print("SKIP : " + name)

        random.shuffle(train_data)
        X, Y = [], []
        for data in train_data:
            X.append(data[0])
            Y.append(data[1])

        test_idx = math.floor(len(X) * 0.8)

        X_train = np.array(X[0:test_idx])
        X_test = np.array(X[test_idx:])
        y_train = np.array(Y[0:test_idx])
        y_test = np.array(Y[test_idx:])

        xy = (X_train, X_test, y_train, y_test)
        np.save("../npy/artists.npy", xy)


if __name__ == "__main__":
    main()
