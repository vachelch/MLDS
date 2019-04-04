from __future__ import print_function
import keras
from keras.datasets import mnist
# from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)

batch_size = 128
num_classes = 10
epochs = 100

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
# print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)



model = Sequential()

model.add(Conv2D(48, kernel_size=(3, 3),
                 activation='relu',padding='same',
                 input_shape=input_shape))
# model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))


# model.add(Conv2D(16, kernel_size=(3, 3),
#                  activation='relu',padding='same',
#                  input_shape=input_shape))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(16, (3, 3),padding='same', activation='relu'))
# model.add(Conv2D(24, (3, 3),padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(20, activation='relu'))
# # model.add(Dropout(0.5))



# model.add(Conv2D(8, kernel_size=(3, 3),
#                  activation='relu',padding='same',
#                  input_shape=input_shape))
# model.add(Conv2D(8, (3, 3),padding='same', activation='relu'))
# # model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(8, (3, 3),padding='same', activation='relu'))
# model.add(Conv2D(8, (3, 3),padding='same', activation='relu'))
# model.add(Conv2D(16, (3, 3),padding='same', activation='relu'))
# # model.add(Conv2D(4, (3, 3),padding='same', activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# # model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(30, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dropout(0.5))


model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=['accuracy'])
model.summary()
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test),shuffle=True)
score = model.evaluate(x_test, y_test, verbose=0)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])

mnist_loss1=history.history['loss']
mnist_acc1=history.history['acc']
# re_val_loss=history.history['val_acc']
# re_val_acc=history.history['val_acc']
mnist_loss1=np.array(mnist_loss1)
mnist_acc1=np.array(mnist_acc1)
# re_val_loss=np.array(re_val_loss)
# re_val_acc=np.array(re_val_acc)
np.save('mnist_acc1.npy',mnist_acc1 )
np.save('mnist_loss1.npy',mnist_loss1 )
# np.save('re_val_loss1.npy', re_val_loss)
# np.save('re_val_acc2.npy', re_val_acc)

# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# preds=model.predict(X_test)
# print (preds)
# model.save('train_mnist6.h5',overwrite=True)
# score = model.predict(x, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])
# plt.plot(x, score)
