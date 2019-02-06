import sys
import os
import numpy as np
import pandas as pd
import gc
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import datetime
import matplotlib.pyplot as plt

# モデル生成
class TrainModel :
  def __init__(self):
    input_dir = 'data'
    #フォルダ数をラベル数とする
    self.nb_classes = len([name for name in os.listdir(input_dir) if name != ".DS_Store"])
    #data_create.pyで作成したデータを読み込む
    x_train, x_test, y_train, y_test = np.load("./npy/artists.npy")
    # 学習データ正規化(0.0-1.0)
    self.x_train = x_train.astype("float") / 255
    self.x_test = x_test.astype("float") / 255
    # 正解データの加工
    self.y_train = np_utils.to_categorical(y_train, self.nb_classes)
    self.y_test = np_utils.to_categorical(y_test, self.nb_classes)

  def train(self, input=None) :
    model = Sequential()
    # K=32, M=3, H=3
    if input == None :
      model.add(Conv2D(32, (3, 3), padding='same',input_shape=self.x_train.shape[1:]))
    else :
      model.add(Conv2D(32, (3, 3), border_mode='same', input_shape=input))

    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(self.nb_classes))
    model.add(Activation('softmax'))
    #モデルをコンパイル
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    if input == None :
      # 学習を実行しモデルを保存
      date_str = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
      history = model.fit(self.x_train, self.y_train, batch_size=32, nb_epoch=30)
      hdf5_file = "./model/artist-model.hdf5"
     # model.save_weights("./model/artist-model-" + date_str + ".hdf5")
      model.save_weights(hdf5_file)

      # モデルのテスト
      score = model.evaluate(self.x_test, self.y_test, verbose=0)
      print('loss=', score[0])
      print('accuracy=', score[1])
      
      #学習過程プロット
      plt.plot(history.history["acc"], label="acc", ls="-", marker="o")
      plt.ylabel("accuracy")
      plt.xlabel("epoch")
      plt.legend(loc="best")
      plt.show()


    return model

if __name__ == "__main__":
  args = sys.argv
  train = TrainModel()
  train.train()
  gc.collect()
