from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


grad=np.load('mnist_hw1-2-2grad.npy')
loss=np.load('mnist_hw1-2-2loss.npy')



line1,=plt.plot(grad,label='gradient')
plt.title('Mnist gradient iteration' )
plt.xlabel('iteration')
plt.ylabel('gradient')
# plt.legend(handles=[line7,line4,line5,line6],loc=1)
# plt.show()
plt.savefig('gradient-iteration.png')


plt.figure()


line2,=plt.plot(loss,label='loss')

plt.title('Mnist loss iteration' )
plt.xlabel('iteration')
plt.ylabel('loss')
# plt.legend(handles=[line7,line4,line5,line6],loc=1)
# plt.show()
plt.savefig('loss-iteration.png')



