# import matplotlib.pyplot as plt
# import numpy as np
# from tensorflow.keras.models import *
# from tensorflow.keras.layers import *
# from tensorflow.keras.datasets import mnist
#
#
#
# input_img = Input(shape = (28,28,1)) # 처음 넣는 이미지 [[1], [2], ....,[28]]인 게 28개인 3차원 이미지의 색까지 숫자로->3차원
# x = Conv2D(16, (3,3), activation='relu', #kernal = (3,3)
#            padding= 'same')(input_img)      #padding= 'same'-> 사이즈 유지 28*28
# x = MaxPooling2D((2,2), padding= 'same')(x)   #필터로 나눈 것 중 남는 것을 쓸지 말지-> padding= 'same' 남는 padding을 쓴다 14*14
# x = Conv2D(8, (3,3), activation='relu',
#            padding= 'same')(x)  #14*14
# x = MaxPooling2D((2,2), padding= 'same')(x) #MaxPooling2D-> 4개를 하나로 줄이는 것 7*7
# x = Conv2D(8, (3,3), activation='relu',
#            padding= 'same')(x) # 7*7
# x = MaxPooling2D((2,2), padding= 'same')(x)
# encoded = MaxPooling2D((2,2), padding= 'same')(x)# #4*4  padding= 'same'을 안 쓰면 3*3
# # decoder는 필터를 반대로 사용해야한다.
#
# x = Conv2D(8, (3,3), activation='relu', padding= 'same')(encoded) #4*4
# x = UpSampling2D((2,2))(x) # MaxPooling2D로 줄인 것을 다시 키우는 것 # 8*8
# x = Conv2D(8, (3,3), activation='relu', padding= 'same')(x) # 8*8
# x = UpSampling2D((2,2))(x)                                         # 16*16
# x = Conv2D(16, (3,3), activation='relu')(x)                     #  padding =same을 안 사용하면 kernal size-1만큼 사이즈는 줄어는다
# x = UpSampling2D((2,2))(x)                                          #28*28
# decoded = Conv2D(1, (3,3), activation='sigmoid',
#                        padding= 'same')(x)                       #28*28
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.datasets import mnist

input_img = Input(shape=(28, 28, 1))
x = Conv2D(16, (3, 3), activation='relu',
           padding='same')(input_img)           # 28 * 28
x = MaxPooling2D((2, 2), padding='same')(x)     # 14 * 14
x = Conv2D(8, (3, 3), activation='relu',
           padding='same')(x)                   # 14 * 14
x = MaxPooling2D((2, 2), padding='same')(x)     # 7 * 7
x = Conv2D(8, (3, 3), activation='relu',
           padding='same')(x)                   # 7 * 7
encoded = MaxPooling2D((2, 2), padding='same')(x) # 4 * 4

x = Conv2D(8, (3, 3), activation='relu',
           padding='same')(encoded)             # 4 * 4
x = UpSampling2D((2, 2))(x)                     # 8 * 8
x = Conv2D(8, (3, 3), activation='relu',
           padding='same')(x)                    # 8 * 8
x = UpSampling2D((2, 2))(x)                     # 16 * 16
x = Conv2D(16, (3, 3), activation='relu')(x)    # 14 * 14
x = UpSampling2D((2, 2))(x)                     # 28 * 28
decoded = Conv2D(1, (3, 3), activation='relu',
           padding='same')(x)                   # 28 * 28

autoencoder = Model(input_img, decoded) # 모델 만들 때 입력(input_img)과 출력만(decoded) 대신 연결디 되게

#autoencoder.summary()
# #파라미터 개수 = 784* 32 +32 +32*784 +784
# encoder = Model(input_img, encoded) # 784(input_img)개를 입력, 32(encoded)개를 출력, 하나의 모델이지만 앞부분만 짤라온 것
# encoder.summary()
#
# encoder_input = Input(shape=(32,))
# decocer_layer = autoencoder.layers[-1]# autoencoder의 마지막, 출력이 나오는 것, autoencoder를 쪼개는 과정
# decoder = Model(encoder_input, decocer_layer(encoder_input)) #encoder_input(32개의 입력) decocer_layer(encoder_input)->32개의 입력을 받아 784개의 출력을
# # autoencoder의 뒷부분을 지정

autoencoder.compile(optimizer='adam', loss = 'binary_crossentropy')
autoencoder.summary()
(x_train, _), (x_test, _) = mnist.load_data() # 입력 후 출력이 나오는 지가 중요, label은 안 중요
x_train = x_train/255 # 색은 255 개이기 때문에 스케일링 작업
x_test = x_test/255

conv_x_train = x_train.reshape(-1, 28 , 28, 1) # 3차원에 맞에 28*28*1로
conv_x_test = x_test.reshape(-1, 28 , 28, 1)
print(conv_x_train.shape)
print(conv_x_test.shape)

fit_hist = autoencoder.fit(
    conv_x_train, conv_x_train,
    epochs = 100, batch_size = 256,
    validation_data = (conv_x_test, conv_x_test)) # 입력 데이터만 있고 라벨이 없기 때문에 비지도 학습, 새로운 그림이 맞았다 틀렸다는 없다.

decoded_img = autoencoder.predict(conv_x_test[:10])


n = 10

plt.figure(figsize=(20,4))
plt.gray()
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

autoencoder.save('./models/autoencoder.h5')