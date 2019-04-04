import torch

from utils import *
from my_models import Generator

import numpy as np 
from skimage import io
from skimage.transform import resize
import os

img_dir = '../../faces/'
data_path = 'resized_imgs.npy'
if os.path.exists(data_path):
    data = np.load(data_path)
    print('load data')
else:
    data = []

    lists = [os.path.join(img_dir, img) for img in os.listdir(img_dir)]
    # lists = ['../../faces/5.jpg']

    for img_path in lists:
        img = io.imread(img_path)
        img_resized = resize(img, (64, 64), mode='constant')
        data.append(img_resized)

    data = np.array(data)
    np.save(data_path, data)

class Dataset():
    def batch(self, batch_size):
        length = len(data)

        while 1:
            batch_idxs = np.random.randint(0, length, batch_size)
            true_imgs = data[batch_idxs].transpose(0, 3, 1, 2)

            yield true_imgs

    def data_generator(self, batch_size):
        self.generator = self.batch(batch_size)
    def next_batch(self):
        return next(self.generator)

































