from keras.datasets import mnist
import tensorflow as tf
from keras.layers.core import Dense
from keras.models import Sequential,load_model
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint,LambdaCallback
from keras.optimizers import RMSprop,adam
import numpy as np
import os
(x_train,y_train),(x_test,y_test)=mnist.load_data()

#print(x_train[1,:].shape)
#數據類型   原來是uint8
#print(x_train.dtype)
#print(np.size(x_train,0),np.size(x_train,1),np.size(x_train,2))
x_train=x_train[0:6000,:,:]
y_train=y_train[0:6000]
x_test=x_test[0:100,:,:]
y_test=y_test[0:100]

x_train=x_train.reshape(np.size(x_train,0),np.size(x_train,1)*np.size(x_train,1))
x_train=x_train.astype("float32")

x_test=x_test.reshape(np.size(x_test,0),np.size(x_test,1)*np.size(x_test,1))
x_test=x_test.astype("float32")
#x_p=x_train[1,:].reshape(28,28)
#print(x_p.shape)
#plt.matshow(x_p)
#plt.show()
x_train/=255
x_test/=255
print(x_train.shape[0],'train samples')
print(x_test.shape[0],'test sample')

#data pre-processing
y_train=np_utils.to_categorical(y_train,num_classes=10)
y_test=np_utils.to_categorical(y_test,num_classes=10)
num=np.array([1,2,3,4,5,6,7,8])
#model
for i in range (1,9):

    model=Sequential()
    model.add(Dense(30,input_shape=(784,),activation=('relu')))
    model.add(Dense(80,activation=('relu')))
    #model.add(Dense(80,activation=('relu')))
    model.add(Dense(10,activation=('softmax')))
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])

#fit the model
    name='model_'+str(num[i-1])
    filepath=name+'weight.{epoch:03d}-loss{loss:.3f}-acc{acc:.3f}.h5'
    checkpoint =ModelCheckpoint(filepath, verbose=1, save_best_only=False, mode='min',period=3)
    his_sh= model.fit(x_train, y_train,batch_size=80, nb_epoch=60,verbose=1, validation_data=(x_test, y_test),callbacks=[checkpoint])
    del model

#score_she = model_sh.evaluate(x_test, y_test, verbose=1)
#test_data=model_sh.predict(x_test,batch_size=100)
"""
model_sh.save('model_sh1.h5')
model_sh.save_weights('model_sh1_weights.h5')

myarray1=np.asarray(weights[0])
print(myarray1)

#name_w='layear'+str(num[1])+'.txt'
#np.savetxt(name_w,myarray1,fmt='%f9')
"""
