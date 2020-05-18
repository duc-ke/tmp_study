import sys # 파이썬 인터프리터 제어
import tensorflow as tf
import keras
from keras.models import Sequential # 모델 레이어를 선형으로 연결하여 구성, 레이어 추가 가능 (add를 이용하여)
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import numpy as np

# 데이터 변환 
## (60000, 28,28) 모양안에 0에서 255까지 범위에 원소로 가득한 행렬 데이터를 0~1 사이 실수로 가득한 (60000,784)모양의 행렬 데이터 변환
### reshape : cnn 구현 할 시에 [pixels][width][height] 형태로 input, pixel은 흑백 1 컬러면 3을 넣기 

img_rows = 28
img_cols = 28

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

input_shape = (img_rows, img_cols, 1) 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols,1) 
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# keras.utils.to_categorical : class vector들을 binary class matrix 형태로 변환 
## y : class vector, num_class : 총 클래스 수 and dtype: 데이터 타입

batch_size = 128
num_classes = 10
epochs = 12
    
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

## Con2D 레이어 
### padding : 경계 처리 방법 (valid: 유효한 영역만 출력, same : 출력 이미지 사이즈와 입력 사이즈가 동일)
### strides : 필터를 움직이는 간격 (좌측 상단 부터 일정한 간격)
## Maxpooling2D : 사소한 변화를 무시해주는 레이어
### pool_size : 수직, 수평 축소 비율 지정 

model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])