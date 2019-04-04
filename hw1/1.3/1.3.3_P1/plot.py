import numpy as np 
import matplotlib.pyplot as plt 

acc = np.load('log/accs.npy')
losses = np.load('log/losses.npy')

test_acc = np.load('log/test_accs.npy')
test_losses = np.load('log/test_losses.npy')

a = np.linspace(-1, 2, 100)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(a, np.log(losses), color = 'r', linestyle='-', label = 'train_loss')
ax1.plot(a, np.log(test_losses), color = 'b', linestyle='--', label = 'test_loss')
ax1.set_ylabel('cross entropy log scale')

ax2 = ax1.twinx()
ax2.plot(a, acc, color = 'r', linestyle='-', label = 'train_acc')
ax2.plot(a, test_acc, color = 'b', linestyle='--', label = 'test_acc')
ax2.set_ylabel('accuracy')
plt.xlabel('alpha')
plt.title('linear interpolation of two model')
plt.legend(labels = ['train', 'test'])
plt.savefig('interpolation.png')
plt.show()



