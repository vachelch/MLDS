from dataset import Dataset
import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

dataset = Dataset()

np.random.seed(0)
r, c = 5, 5

dataset.data_generator(64)
true_imgs = dataset.next_batch()
true_imgs = true_imgs.transpose(0, 2, 3, 1)
fig, axs = plt.subplots(r, c)
cnt = 0
for i in range(r):
    for j in range(c):
        axs[i,j].imshow(true_imgs[cnt, :,:,:])
        axs[i,j].axis('off')
        cnt += 1
fig.savefig('log/test.png')
plt.close()