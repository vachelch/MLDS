import numpy as np 
import matplotlib.pyplot as plt

dqn = np.load('dqn_test_history.npy')
ddqn = np.load('ddqn_test_history.npy')

plt.figure(figsize = (12, 7))
plt.plot(dqn[:, 1], dqn[:, 0], label = 'DQN')
plt.title('learning curve of DQN')
plt.xlabel('epoch')
plt.ylabel('average clipped reward')
plt.legend()
plt.savefig('dqn.png')


plt.figure(figsize = (12, 7))
plt.plot(ddqn[:, 1], ddqn[:, 0], color = 'C1', label = 'Double DQN')
plt.title('learning curve of Double DQN')
plt.xlabel('epoch')
plt.ylabel('average clipped reward')
plt.legend()
plt.savefig('ddqn.png')


plt.figure(figsize = (12, 7))
plt.plot(dqn[:, 1], dqn[:, 0], label = 'DQN')
plt.plot(ddqn[:, 1], ddqn[:, 0], label = 'Double DQN', color = 'C1')
plt.title('learning curves')
plt.xlabel('epoch')
plt.ylabel('average clipped reward')
plt.legend()
plt.savefig('dqn_and_ddqn.png')


