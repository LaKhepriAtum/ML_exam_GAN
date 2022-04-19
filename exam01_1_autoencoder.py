import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist



input_img = Input(shape = (784,)) # 처음 넣는 이미지
encoded = Dense(128, activation='relu')(input_img)# 줄이는 과정을 따로 시작한다, 지금까지는 sequencial 에 add를 했지만, input 까지 주기
encoded = Dense(64, activation='relu')(encoded) # 입력은 encoded로 받아서 encoded에 덮어쓰기
encoded = Dense(32, activation='relu')(encoded) # 입력은 encoded로 받아서 encoded에 덮어쓰기
decoded = Dense(64, activation='sigmoid')(encoded) # 0~ 1 사이의 값 출력 , 784층이 있다
decoded = Dense(128, activation='sigmoid')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)
autoencoder = Model(input_img, decoded) # 모델 만들 때 입력(input_img)과 출력만(decoded) 대신 연결디 되게

# autoencoder.summary()
# #파라미터 개수 = 784* 32 +32 +32*784 +784
# encoder = Model(input_img, encoded) # 784(input_img)개를 입력, 32(encoded)개를 출력, 하나의 모델이지만 앞부분만 짤라온 것
# encoder.summary()
#
# encoder_input = Input(shape=(32,))
# decocer_layer = autoencoder.layers[-1]# autoencoder의 마지막, 출력이 나오는 것, autoencoder를 쪼개는 과정
# decoder = Model(encoder_input, decocer_layer(encoder_input)) #encoder_input(32개의 입력) decocer_layer(encoder_input)->32개의 입력을 받아 784개의 출력을
# # autoencoder의 뒷부분을 지정

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data() # 입력 후 출력이 나오는 지가 중요, label은 안 중요
x_train = x_train/255 # 색은 255 개이기 때문에 스케일링 작업
x_test = x_test/255

flatten_x_train = x_train.reshape(-1, 28 * 28)
flatten_x_test = x_test.reshape(-1, 28 * 28)
print(flatten_x_train.shape)
print(flatten_x_test.shape)

fit_hist = autoencoder.fit(
    flatten_x_train, flatten_x_train,
    epochs = 100, batch_size = 256,
    validation_data = (flatten_x_test, flatten_x_test)) # 입력 데이터만 있고 라벨이 없기 때문에 비지도 학습, 새로운 그림이 맞았다 틀렸다는 없다.

decoded_img = autoencoder.predict(flatten_x_test[:10])

n = 10
plt.gray()
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2,10, i+1)
    plt.imshow(x_test[i]) # 원본 x_test 출력
    ax.get_xaxis().set_visible(False) #눈금 제거
    ax.get_yaxis().set_visible(False) #눈금 제거

    ax = plt.subplot(2, 10, i+1+n)
    plt.imshow(decoded_img[i].reshape(28, 28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

plt.plot(fit_hist.history['loss'])
plt.plot(fit_hist.history['val_loss'])
plt.show()