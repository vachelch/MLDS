import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt

data=np.load('hw1-2-3-666.npy')

plt.xlabel('minimum_ratio')
plt.ylabel('loss')

plt.scatter(data[0:100,2], data[0:100,0])
plt.savefig('loss-minimum_ratio.png')
# line2, = plt.plot(get_gradient.loss, color='m', label='loss')
# plt.ylim(0,1)
# plt.yticks(np.linspace(0,1,11))
# pit2.tick_params(axis='y')
# plt.legend(handles=[line1,line2])
# plt.show()