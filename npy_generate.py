from PIL import Image
import os
import glob
import numpy as np
import math
import random

image_size = 50
num_traindata = 500
Image.LOAD_TRUNCATED_IMAGES = True


def main():
    train_data = []
    artistname = []
    input_dir = "images"
    dir_list = os.listdir(input_dir)
    for dir_name in dir_list:
        if dir_name == "upload_images":
            continue
        artistname.append(dir_name)

    for name_idx, name in enumerate(artistname):
        train_cnt = 0
        try:
            print("---", name)
            image_dir = input_dir + "/" + name
            files = glob.glob(image_dir + "/*.jpg")
            for file_idx, file in enumerate(files):
                if train_cnt >= num_traindata:
                    break
                img = Image.open(file)
                img = img.convert("RGB")
                img = img.resize((image_size, image_size))
                # rotation
                for angle in range(-20, 20, 5):
                    img_r = img.rotate(angle)
                    img_np = np.asarray(img_r)
                    train_data.append([img_np, name_idx])
                    train_cnt += 1
                    # inverse
                    img_trans = img_r.transpose(Image.FLIP_LEFT_RIGHT)
                    img_np = np.asarray(img_trans)
                    train_data.append([img_np, name_idx])
                    train_cnt += 1
                    if train_cnt >= num_traindata:
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
        np.save("./npy/artists.npy", xy)


if __name__ == "__main__":
    main()
