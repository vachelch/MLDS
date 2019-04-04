
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
from keras.callbacks import ModelCheckpoint

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
# config.gpu_options.per_process_gpu_memory_fraction = 0.2
session = tf.Session(config=config)

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
# print(x_train.shape[0], 'train samples')
# print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print('y_train shape:', y_train.shape)

input_shape=x_train.shape[1:]




def get_gradient_norm_func(model):
    grads = K.gradients(model.total_loss, model.trainable_weights)
    summed_squares = [K.sum(K.square(g)) for g in grads]
    norm = K.sqrt(sum(summed_squares))
    inputs = model.model._feed_inputs + model.model._feed_targets + model.model._feed_sample_weights
    # input_tensors = [model.inputs[0],  # input data
    #                  model.sample_weights[0],  # how much to weight each sample by
    #                  model.targets[0],  # labels
    #                  K.learning_phase(),  # train or test mode
    #                  ]

    func = K.function(inputs, [norm])
    return func

model = Sequential()
model.add(Flatten(input_shape=input_shape))
model.add(Dense(128, activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# history = LossHistory()
# history = get_gradient_norm_func(model)

# callback=ModelCheckpoint('mnist_model8_{epoch:d}.h5', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=3)
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
model.summary()
grad=[]
loss=[]
for ep in range(epochs):
    for iter in range(300):
        # print(ep,iter)
        x=x_train[iter*200:(iter+1)*200][:][:]
        y = y_train[iter * 200:(iter + 1) * 200][:]
        his=model.fit(x, y,
                  batch_size=200,
                  epochs=1,
                  verbose=0)
        get_gradient = get_gradient_norm_func(model)
        g=get_gradient([x,y,np.ones(len(y))])
        # print(g)
        grad.append(g)
        loss.append(his.history['loss'])
        # del get_gradient
        # del g


# print(grad)

# de=np.array(get_gradient.losses)
# print(de)
#
# print(len(get_gradient.grad5))
# for i in range (len(get_gradient.gradients)):
#     print(get_gradient.grad5[i]([x_train,y_train,np.ones(len(y_train))]))

# plt.xlabel('iteration')
# plt.ylabel('gradient')
# plt.show()

# ine1,=plt.plot(history.history['acc'],label='train_acc')
# line2,=plt.plot(history.history['val_acc'],label='val_acc')

# plt.ylim(0,1)

# plt.yticks(np.linspace(0,1,21))
# plt.title('The accuracy and loss of CNN classifier')
# plt.ylabel('loss')
# plt.xlabel('epoch')

# plt.legend(['train_acc', 'test_acc'], loc=7)

# plt2=plt.twinx()
# plt2.set_ylabel('loss')

# line1, = plt.plot(grad, color='red', label='grad')

# line2, = plt.plot(loss, color='m', label='loss')
# plt.ylim(0,1)
# plt.yticks(np.linspace(0,1,11))
# pit2.tick_params(axis='y')
# plt.legend(handles=[line1,line2])
# plt.show()


grad=np.array(grad)
loss=np.array(loss)
np.save('mnist_hw1-2-2grad.npy',grad )
np.save('mnist_hw1-2-2loss.npy',loss )


