import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

autoencoder = load_model('./models/autoencoder_noisy.h5')

(x_train, _), (x_test, _) = mnist.load_data() # 입력 후 출력이 나오는 지가 중요, label은 안 중요

x_train = x_train / 255 # 색은 255 개이기 때문에 스케일링 작업
x_test = x_test / 255

conv_x_train = x_train.reshape(-1, 28, 28, 1) # 3차원에 맞에 28*28*1로
conv_x_test = x_test.reshape(-1, 28, 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

noise_factor = 0.5
conv_x_test_noisy = conv_x_test + np.random.normal(  # 평균이 0, 표준편차가 1
    loc=0.0, scale=1.0, size=conv_x_test.shape) * noise_factor # 값이 절반으로
conv_x_test_noisy = np.clip(conv_x_test_noisy, 0.0, 1.0) #clip->  0.0, 1.0기준으로 값을 짜른다. 0보다 작은 값은 0 1보다 큰 값은 14

decoded_img = autoencoder.predict(conv_x_test_noisy[:10]) #  잡음이 들어간 data를 넣는다

n = 10

plt.figure(figsize=(20, 4))
plt.gray()
for i in range(n):
    ax = plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i])  # 원본 x_test 출력
    ax.get_xaxis().set_visible(False) #눈금 제거
    ax.get_yaxis().set_visible(False) #눈금 제거

    ax = plt.subplot(3, 10, i + 1 + n)
    plt.imshow(conv_x_test_noisy[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, 10, i + 1 + n * 2)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

