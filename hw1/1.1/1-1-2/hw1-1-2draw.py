from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np

mnist_loss1=np.load('mnist_loss1.npy')
mnist_loss2=np.load('mnist_loss2.npy')
mnist_loss3=np.load('mnist_loss3.npy')
mnist_acc1=np.load('mnist_acc1.npy')
mnist_acc2=np.load('mnist_acc2.npy')
mnist_acc3=np.load('mnist_acc3.npy')
# mnist_loss4=np.load('mnist_loss4.npy')
# mnist_loss5=np.load('mnist_loss5.npy')
# mnist_loss6=np.load('mnist_loss6.npy')
# mnist_acc4=np.load('mnist_acc4.npy')
# mnist_acc5=np.load('mnist_acc5.npy')
# mnist_acc6=np.load('mnist_acc6.npy')
# s = 1000
# f = 5
# sample = 1000
# x = np.arange(sample)
# x=x/1000
# # np.random.shuffle(x)
# # print(x[0:10])
# y = x*np.sin(2 * np.pi * x)+x*np.cos(2 * np.pi * x*5)/(2)
# epoch=np.arange(10000)
# epoch2=np.arange(100)
epoch3=np.arange(50)
# sqwave = np.sign(np.sin(2*np.pi*f*x))

line1,=plt.plot(epoch3, mnist_acc1,label='1_hid_layer')
line2,=plt.plot(epoch3, mnist_acc2,label='3_hid_layer')
line3,=plt.plot(epoch3, mnist_acc3,label='5_hid_layer')

# line4,=plt.plot(epoch3, mnist_acc4,label='1')
# line5,=plt.plot(epoch3, mnist_acc5,label='2')
# line6,=plt.plot(epoch3, mnist_acc6,label='3')
# plt.plot(epoch, sqwave)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.ylim(0,1)
# plt.xlim(0,50)
plt.title('1-1-2Mnist  cnn model accuracy' )
plt.legend(handles=[line1,line2,line3],loc=0)
plt.savefig('1-1-2Mnist  cnn model accuracy' )
# plt.show()

plt.figure()

line1,=plt.plot(epoch3, mnist_loss1,label='1_hid_layer')
line2,=plt.plot(epoch3, mnist_loss2,label='3_hid_layer')
line3,=plt.plot(epoch3, mnist_loss3,label='5_hid_layer')

# line4,=plt.plot(epoch3, mnist_acc4,label='1')
# line5,=plt.plot(epoch3, mnist_acc5,label='2')
# line6,=plt.plot(epoch3, mnist_acc6,label='3')
# plt.plot(epoch, sqwave)
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim(0,1)
# plt.xlim(0,50)
plt.title('1-1-2Mnist  cnn model loss ' )
plt.legend(handles=[line1,line2,line3],loc=0)
plt.savefig('1-1-2Mnist  cnn model loss ' )
# plt.show()