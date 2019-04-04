from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D,ZeroPadding2D
from keras.utils import np_utils
import tensorflow as tf
from keras import optimizers
# from sklearn.metrics import confusion_matrix
from keras import backend as K
# from keras.layers.normalization import BatchNormalization
# from keras import regularizers
# from keras.backend.tensorflow_backend import set_session
# from keras.preprocessing.image import ImageDataGenerator

K.set_learning_phase(0)

np.random.seed(1337)


Fs = 1000
f = 5
sample = 1000
x = np.arange(sample)
x=x/1000
# np.random.shuffle(x)
# print(x[0:10])
y = x*np.sin(2 * np.pi * x)+x*np.cos(2 * np.pi * x*10)/(2)


# x2=np.arange(sample)

y2=np.sign(np.sin(2*np.pi*5*x))


# print(x[0:10])
# print(y[0:10])
# plt.plot(x, y2)
# plt.xlabel('sample(n)')
# plt.ylabel('voltage(V)')
# plt.show()



batch_size = 128
# nb_classes = 7
nb_epoch = 10000

X_train = x.astype('float32')
# X_test = X_test.astype('float32')

# print('X_train shape:', X_train.shape)
# print(X_train.shape[0], 'train samples')
# print(X_test.shape[0], 'test samples')
# Y_train=y.astype('float32')
Y_train=y.astype('float32')

model = Sequential()
# users=Input(shape=[1])

# model.add(Flatten())

# model.add(Dense(10,input_shape=(1,),activation='relu'))
# # model.add(Dense(10,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(20,activation='relu'))
# model.add(Dense(15,activation='relu'))
# model.add(Dense(8,activation='relu'))


model.add(Dense(20,input_shape=(1,),activation='relu'))
model.add(Dense(25,activation='relu'))
model.add(Dense(20,activation='relu'))
model.add(Dense(20,activation='relu'))



# model.add(Dense(370,input_shape=(1,),activation='relu'))
# # # model.add(Dense(1000,activation='relu'))


model.add(Dense(1,activation='linear'))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(nb_classes))
# model.add(Activation('softmax'))

model.compile(loss='mse',
              optimizer='adam')
# print(model.summary())

history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,shuffle=True,
          verbose=2)

re_loss=history.history['loss']
# re_acc=history.history['acc']
# re_val_loss=history.history['val_acc']
# re_val_acc=history.history['val_acc']
re_loss=np.array(re_loss)
# re_acc=np.array(re_acc)
# re_val_loss=np.array(re_val_loss)
# re_val_acc=np.array(re_val_acc)
# np.save('re_acc2.npy',re_acc )
np.save('re_loss2.npy',re_loss )
# np.save('re_val_loss1.npy', re_val_loss)
# np.save('re_val_acc2.npy', re_val_acc)

# score = model.evaluate(X_test, Y_test, verbose=0)
# print('Test score:', score[0])
# print('Test accuracy:', score[1])

# preds=model.predict(X_test)
# print (preds)
model.save('train2.h5',overwrite=True)
