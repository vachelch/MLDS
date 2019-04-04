import numpy as np

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt 

import torch
from torch.autograd import Variable

# def scale(img):
#     img_ = np.copy(img.astype('float32'))
#     img_ -= np.min(img_)
#     img_ /= np.max(img_)
#     return img_

# def descale(img):
#     img_ = np.copy(img)
#     img_ -= np.min(img_)
#     img_ /= np.max(img_)
#     img_ = (img_ * 255).astype(np.uint8)
#     return img_

def save_imgs(generator):
    np.random.seed(0)
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator.predict(noise)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig('../../samples/gan.png')
    plt.close()

def Logger(generator, filename):
    np.random.seed(0)
    r, c = 5, 5
    noise = Variable(torch.randn(r * c, 100).cuda())
    # gen_imgs should be shape (25, 64, 64, 3)
    gen_imgs = generator(noise)
    gen_imgs = gen_imgs.cpu().data.numpy().transpose(0, 2, 3, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig('log/' + filename)
    plt.close()









