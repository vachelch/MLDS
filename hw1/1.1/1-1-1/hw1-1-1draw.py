from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model

re_loss1=np.load('re_loss1.npy')
re_loss2=np.load('re_loss2.npy')
re_loss3=np.load('re_loss3.npy')
# re_loss4=np.load('re_loss4.npy')
# re_loss5=np.load('re_loss5.npy')
# re_loss6=np.load('re_loss6.npy')


s = 1000
f = 5
sample = 1000
x = np.arange(sample)
x=x/1000
# np.random.shuffle(x)
# print(x[0:10])
y = x*np.sin(2 * np.pi * x)+x*np.cos(2 * np.pi * x*10)/(2)
y2=np.sign(np.sin(2*np.pi*5*x))
epoch=np.arange(10000)
epoch2=np.arange(20000)
epoch3=np.arange(1000)
# sqwave = np.sign(np.sin(2*np.pi*f*x))

line1,=plt.plot(epoch, re_loss1,label='1_hid_layer')
line2,=plt.plot(epoch, re_loss2,label='3_hid_layer')
line3,=plt.plot(epoch, re_loss3,label='5_hid_layer')
# plt.title(r'$f(x) =xsin(2\pi x)+xcos(20\pi x)/2$' + ' Model loss' )
# plt.title(r'$f(x) =sgn(sin(10\pi x))$' + ' Model loss' )
plt.title(r'$f(x) =xsin(2\pi x)+xcos(20\pi x)/2$' + ' loss' )
# plt.title(r'$f(x) =sgn(sin(10\pi x))$' + ' Simulatation' )


plt.xlabel('iteration')
plt.ylabel('loss')
plt.legend(handles=[line1,line2,line3],loc=1)
# plt.show()
plt.savefig('1-1-1loss.png')

plt.figure()

# plt.plot(epoch, sqwave)
# plt.title('xsin(2pix)+xcos(10pix)/2  Model loss')
# (np.sin(2*np.pi*5*x))
loaded_model1=load_model('train1.h5')
loaded_model2=load_model('train2.h5')
loaded_model3=load_model('train3.h5')
score1 = loaded_model1.predict(x, verbose=0)
score2 = loaded_model2.predict(x, verbose=0)
score3 = loaded_model3.predict(x, verbose=0)

line7,=plt.plot(epoch3, y,label='original')
line4,=plt.plot(epoch3, score1,label='1_hid_layer')
line5,=plt.plot(epoch3, score2,label='3_hid_layer')
line6,=plt.plot(epoch3, score3,label='5_hid_layer')


# plt.title(r'$f(x) =xsin(2\pi x)+xcos(20\pi x)/2$' + ' Model loss' )
# plt.title(r'$f(x) =sgn(sin(10\pi x))$' + ' Model loss' )
plt.title(r'$f(x) =xsin(2\pi x)+xcos(20\pi x)/2$' + ' Simulatation' )
# plt.title(r'$f(x) =sgn(sin(10\pi x))$' + ' Simulatation' )


plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend(handles=[line7,line4,line5,line6],loc=1)
# plt.show()
plt.savefig('1-1-Simulatation.png')