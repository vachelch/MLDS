import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

from my_models import Generator, Discriminator
from dataset import Dataset
from utils import *

import os

if not os.path.isdir('log'):
    os.mkdir('log')

D_step = 5
G_step = 1
LR = 2e-4
n_iter = 200000
print_every = 200
batch_size = 64
noise_dim = 100

G_path = 'log/generator.pt'
D_path = 'log/discriminator.pt'

dataset = Dataset()

generator = Generator()
discriminator = Discriminator()
generator = generator.cuda()
discriminator = discriminator.cuda()

# generator.load_state_dict(torch.load(G_path))
# discriminator.load_state_dict(torch.load(D_path))

gen_optim = optim.Adam(generator.parameters(), lr = LR)
discrim_optim = optim.Adam(discriminator.parameters(), lr= LR)

criterion = nn.BCELoss()

def train_D(true_imgs, gen_imgs):
    discrim_optim.zero_grad()
    true_pred = discriminator(true_imgs)
    gen_pred = discriminator(gen_imgs)

    loss_D = criterion(true_pred, Variable(torch.ones(batch_size).cuda()))
    loss_D.backward()

    loss_G = criterion(gen_pred, Variable(torch.zeros(batch_size).cuda()))
    loss_G.backward()

    discrim_optim.step()

    return loss_G + loss_D

def train_G(gen_imgs):
    gen_optim.zero_grad()
    gen_pred = discriminator(gen_imgs)

    loss_G = criterion(gen_pred, Variable(torch.ones(batch_size).cuda()))
    loss_G.backward()

    gen_optim.step()

    return loss_G

def gen_img(generator, batch_size=64, noise_dim = 100):
    noise = Variable(torch.randn(batch_size, noise_dim).cuda())
    gen_imgs = generator(noise)

    return gen_imgs



dataset.data_generator(batch_size)
for i_iter in range(n_iter):
    true_imgs = dataset.next_batch()
    true_imgs = Variable(torch.FloatTensor(true_imgs).cuda())

    loss_D = 0
    for i in range(D_step):
        gen_imgs = gen_img(generator, batch_size)
        loss_D += train_D(true_imgs, gen_imgs)

    loss_G = 0
    for i in range(G_step):
        gen_imgs = gen_img(generator, batch_size)
        loss_G += train_G(gen_imgs)
    loss_D /= D_step
    loss_G /= G_step

    if i_iter % print_every == 0:
        Logger(generator, str(i_iter) + '.png')
        print('iter {}, G_loss: {:0.3f}, D_loss: {:0.3f}'.format(i_iter, loss_G.cpu().data[0], loss_D.cpu().data[0]))
        with open('log/log.txt', 'w+') as f:
            f.write('iter {}, G_loss: {:0.3f}, D_loss: {:0.3f}\n'.format(i_iter, loss_G.cpu().data[0], loss_D.cpu().data[0]))
    if i_iter % 10000 == 0:
        torch.save(generator.state_dict(), 'log/generator_' + str(i_iter) + '.pt')
        torch.save(discriminator.state_dict(), 'log/discriminator_' + str(i_iter) + '.pt')





