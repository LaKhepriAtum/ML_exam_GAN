import matplotlib.pyplot as plt
import numpy as np
import os

from IPython.core.pylabtools import figsize
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import *
from tensorflow.keras.models import *

OUT_DIR = './CNN_out'
img_shape = (28, 28, 1) # 이미지 shape.
epochs = 100000
batch_size = 128
noise = 100
sample_interval =100

(X_train, Y_train), (_, _) = mnist.load_data()
for i in range(10):
  MY_NUMBER = i
  X_train = X_train[Y_train == MY_NUMBER]

  X_train = X_train / 127.5 -1 #색개수 255의 절반 -> 범위가 -1 ~ 1 사이, 색이 옅은 것은 덜 학습하고 큰 값에 민감하게 학습
  X_train = np.expand_dims(X_train, axis = 3) # 60000은 이미지가 60000개 있다는 것,2차원을 3차원으로 차원을 하나 늘리는 것
  print(X_train.shape)

  generator = Sequential()
  generator.add(Dense(256*7*7, input_dim = noise)) # 7*7 256장
  generator.add(Reshape((7, 7, 256))) # shape -> tuple 로 준다
  generator.add(Conv2DTranspose(128, kernel_size=3,
                                strides= 2, padding = 'same')) # conv를 늘려나가기 때문에, strides 2칸씩 건너 띈다(사이즈가 줄어든다)
  generator.add(BatchNormalization()) #BatchNormalization-> 필터로 만든 이미지들을 합치면 숫자가 너무 커지기 때문에 다시 한번 scaling
  generator.add(LeakyReLU(alpha = 0.01)) # 따로 alpha를 주기 위해서는 따로 add를 만들어야 한다.
  generator.add(Conv2DTranspose(64, kernel_size=3,
                                strides= 1, padding = 'same')) # conv를 늘려나가기 때문에, strides 2칸씩 건너 띈다
  generator.add(BatchNormalization()) #BatchNormalization-> 필터로 만든 이미지들을 합치면 숫자가 너무 커지기 때문에 다시 한번 scaling
  generator.add(LeakyReLU(alpha = 0.01))
  generator.add(Conv2DTranspose(1, kernel_size= 3,
                                strides=2, padding = 'same'))
  generator.add(Activation('tanh'))

  discriminator = Sequential()
  discriminator.add(Conv2D(32, kernel_size = 3, # (3,3)을 줄여서 쓴 것
                          strides=2, padding = 'same', input_shape=img_shape))
  discriminator.add(LeakyReLU(alpha=0.01))
  discriminator.add(Conv2D(64, kernel_size= 3,
                          strides = 2, padding = 'same'))
  discriminator.add(LeakyReLU(alpha=0.01))
  discriminator.add(Conv2D(128, kernel_size= 3,
                          strides = 2, padding = 'same'))
  discriminator.add(LeakyReLU(alpha=0.01))
  discriminator.add(Flatten())
  discriminator.add(Dense(1, activation='sigmoid')) # 새로운 그림을 그려 내는지 못하는지를 판별하는 것, 0.3만큼 새롭다 이런 건 없다.
  discriminator.summary()

  discriminator.compile(loss = 'binary_crossentropy',
                        optimizer='adam', metrics=['accuracy'])

  discriminator.trainable=False # discriminator 를 처음에는 하지 않는다
  gan_model = Sequential()
  gan_model.add(generator)
  gan_model.add(discriminator)
  gan_model.summary()
  gan_model.compile(loss = 'binary_crossentropy', optimizer='adam')
  # gan 모델을 학습시킬 때의 값(0,1)을 가지고 generator를 학습,-> generator는 따로 compile 이 없다.

  real = np.ones((batch_size, 1)) # batch_size만큼 1로 채운다 , 1=> 1차원 ->shape

  fake = np.zeros((batch_size, 1)) # batch_size만큼 0로 채운다 , 1=> 1차원

  for epoch in range(epochs):
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      # X_train.shape[0]의 개수 60000개, 0~60000의 무작위 값, batch_size개 만큼-> 중복이 가능하게 무작위로 여러 사이즈로 뽑는 것= 부트스트랩
      # 부트스트랩으로 통계해석을 하는 것=> 배깅
      real_imgs = X_train[idx]

      z = np.random.normal(0, 1, (batch_size, noise)) # 정규분포를 따르는 모양 평균이 0 표준편차가 1, 100 사이즈의 data 가 128개
      fake_imgs = generator.predict(z) # generator predict 해서 만든다

      d_hist_real = discriminator.train_on_batch(real_imgs, real) # train_on_batch data 한 묶음을 주고 학습하고 끝  real_imgs, 라밸 = real_imgs
      d_hist_fake = discriminator.train_on_batch(fake_imgs, fake)
      # real 이미지 128개, fake 이미지 128개 학습 총 256개 학습

      d_loss, d_acc = 0.5 * np.add(d_hist_real, d_hist_fake) # d_hist_real, d_hist_fake의 평균을 내야한다
      discriminator.trainable = False # forward, predict은 가능하지만 backward, 학습은 안 됨

      z = np.random.normal(0, 1, (batch_size, noise)) # 무작위로 새로 생성
      gan_hist = gan_model.train_on_batch(z,real) # 잡음을 주면 generator가 이미지 생성->discriminator가 0, 1 판단 -> 그 data를 가지고 backward(학습) real이 나고게끔

      if epoch % sample_interval ==0:
          print('%d [D loss: %f, acc.: %.2f%%] [G loss: %f]'%(
              epoch, d_loss, d_acc *100, gan_hist
          ))# %d epoch
          row = col = 4
          z = np.random.normal(0, 1, (row * col, noise)) # noise를 4 * 4 개 만든다
          fake_imgs = generator.predict(z)
          fake_imgs = 0.5 * fake_imgs +0.5
          _, axs = plt.subplots(row, col, figsize=(row, col),
                              sharey = True, sharex = True)
          count = 0
          for i in range(row):
              for j in range(col):
                  axs[i, j].imshow(fake_imgs[count, :, :, 0], cmap = 'gray')
                  axs[i, j].axis('off')
                  count +=1
          plt.show()
          generator.save('/content/drive/MyDrive/GAN_modelgenerator_mnist_{}.h5'.format(MY_NUMBER))
          plt.close() # plt 닫아주기, 안 그래면 계속 겹친다