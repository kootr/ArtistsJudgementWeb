from PIL import Image
import sys
import os, glob
import numpy as np
import random, math

#iwasawa_test_20191121
class DataCreate :
  def __init__(self, script_name):
    Image.LOAD_TRUNCATED_IMAGES = True

  def create(self) :
    input_dir = "data"
    artistname = []
    #/data配下の画家ごとに分けられたフォルダ名を配列に渡し、インデックス番号をつける
    dir_list = os.listdir(input_dir)
    for index, dir_name in enumerate(dir_list):
      if dir_name == '.DS_Store' :
        continue
      #画家ごとのフォルダ名を配列に渡す
      artistname.append(dir_name)

    # 画像データとラベルデータ
    image_size = 50
    train_data = []
    #画像を配列に変換(50×50pixel,RGB)
    for idx, name in enumerate(artistname):
      try :
        print("---", name)
        image_dir = input_dir + "/" + name
        files = glob.glob(image_dir + "/*.jpg")
        for i, f in enumerate(files):
          img = Image.open(f)
          img = img.convert("RGB")
          img = img.resize((image_size, image_size))
          data = np.asarray(img)
          train_data.append([data, idx])

      except:
        print("SKIP : " + name)

    # データをshuffle
    random.shuffle(train_data)
    #画像データをX,ラベルをYに代入
    X, Y = [],[]
    for data in train_data:
      X.append(data[0])
      Y.append(data[1])

    test_idx = math.floor(len(X) * 0.8)
    xy = (np.array(X[0:test_idx]), np.array(X[test_idx:]),
          np.array(Y[0:test_idx]), np.array(Y[test_idx:]))
    #結果を保存
    np.save("./npy/artists", xy)

if __name__ == "__main__":
  args = sys.argv
  datacreate = DataCreate(args[0])
  datacreate.create()
