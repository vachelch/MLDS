import numpy as np 
import matplotlib.pyplot as plt 

x = np.load('log/epoch.npy')
loss = np.load('log/losses.npy')
test_loss = np.load('log/test_losses.npy')

plt.figure()
l1 = plt.plot(x, loss, color = 'red', label = 'train')
l2 = plt.plot(x, test_loss, color = 'blue', label = 'test')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

plt.savefig('DNN_fit_random.png')
plt.show()