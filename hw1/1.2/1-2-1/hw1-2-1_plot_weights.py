from keras.models import Sequential,load_model
import matplotlib.pyplot as plt
import h5py
import os
from os import listdir
import numpy as np


#score_she = model_sh.evaluate(x_test, y_test, verbose=1)
#test_data=model_sh.predict(x_test,batch_size=100)

layer_dense='dense_'
mypath="D:\ZT\class\MLDS\hw2\hw1-2-1\optimization process_all"
col='b','r','g','c','m','y','k'
files = listdir(mypath)
print(files[1:11][31:36])
"""
for f in files:
    files_name = os.path.splitext(f[:])[-1]
    if files_name == '.h5':
        if files_h5==[]:
           files_h5=f
        elif files_h5 != []:
            files_h5=np.vstack((files_h5,f))
"""

#read weights
acc=[]
w1=[]
b=[]
for times in range(1,9):  #training 次數
    print(times)

    a=(times-1)*20  #30個Epoch 3個存一筆
    #layear_name=layear_dense+str(1+4*(times-1))

    print(1+3*(times-1))
    for f in range(1,21):    #Epoch 個數
        #print(layear_name)
        model=load_model(files[a+f])

        weights = np.array([[]])   #所有weights
        for d in range(1,4):  #  隱藏層weights  weights=m1n1+m2n2+....
            model.summary()
            all_weights=model.get_layer(layer_dense+str(d+3*(times-1))).get_weights()
            l_w=all_weights[0].reshape(1,np.size(all_weights[0],0)*np.size(all_weights[0],1))
            weights=np.hstack((weights,np.array(l_w)))
        if f==1:
            w1=weights
            acc=np.array(float(files[a + f][31:36]))
        elif f!=1:
           w1=np.vstack((w1,weights))
           acc = np.vstack((acc,np.array(float(files[a + f][31:36]))))
    print(w1.shape)
    if times == 1:
        b = w1
        ww=w1[0:20,0:w1.shape[1]]
        u, s, v = np.linalg.svd(ww)
        S = np.zeros((20, w1.shape[1]))
        S[:2, :2] = np.diag(s[0:2])
        data = np.dot(u, np.dot(S, v))
        #acc=acc.reshape(1,10)
        for i in range(0,20):
            c = float(acc[i])
            plt.plot(data[i, 0], data[i, 1])
            plt.text(data[i, 0], data[i, 1], '%.3f' %c, fontsize=10,color=[0.1,0.5,0.3])

    elif times != 1:
        b = np.concatenate((b, w1))
        ww = b[0+a:20+a, 0:w1.shape[1]]
        u, s, v = np.linalg.svd(ww)
        S = np.zeros((20,w1.shape[1]))
        S[:2, :2] = np.diag(s[0:2])
        data = np.dot(u, np.dot(S, v))
        for i in range(0,20):
            c = float(acc[i])
            plt.plot(data[i, 0], data[i, 1])
            plt.text(data[i, 0], data[i, 1], '%.3f' % c, fontsize=10,color=col[times-2])
        #print(b.size)


"""
myarray1=np.asarray(b)
print(myarray1)

name_w='layear'+str(1)+'.txt'
np.savetxt(name_w,myarray1,fmt='%f9')
"""
plt.xlabel('whole model')
plt.show()

"""
"""
