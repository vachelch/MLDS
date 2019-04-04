import numpy as np 
import matplotlib.pyplot as plt 

acc = np.load('log/accs.npy')
losses = np.load('log/losses.npy')

test_acc = np.load('log/test_accs.npy')
test_losses = np.load('log/test_losses.npy')

sensi = np.load('log/sensi.npy')
test_sensi = np.load('log/test_sensi.npy')

lrs = np.load('log/lrs.npy')

print(lrs)
print(test_sensi)

#loss sensitivity
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(lrs, losses, color = 'b', linestyle='-', label = 'train_loss')
ax1.plot(lrs, test_losses, color = 'b', linestyle='--', label = 'test_loss')
ax1.set_ylabel('loss')
ax1.legend(loc=0)

ax2 = ax1.twinx()
ax2.plot(lrs, sensi, color = 'r', linestyle='-', label = 'sensitivity')
ax2.set_ylabel('sensitivity')
plt.xlabel('learning rate')
plt.title('loss, sensitivity vs lr')
plt.legend()
plt.xticks(np.arange(0.001, 0.006, step = 0.001))
plt.savefig('loss_sensi.png')
plt.show()

#acc sensitivity
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(lrs, acc, color = 'b', linestyle='-', label = 'train_acc')
ax1.plot(lrs, test_acc, color = 'b', linestyle='--', label = 'test_acc')
ax1.set_ylabel('accurcy')
ax1.legend(loc=0)

ax2 = ax1.twinx()
ax2.plot(lrs, sensi, color = 'r', linestyle='-', label = 'sensitivity')
ax2.set_ylabel('sensitivity')
plt.xlabel('learning rate')
plt.title('accuracy, sensitivity vs lr')
plt.legend()
plt.xticks(np.arange(0.001, 0.006, step = 0.001))
plt.savefig('acc_sensi.png')
plt.show()
















