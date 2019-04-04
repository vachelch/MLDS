import torch 
from torch.autograd import Variable
import torch.nn as nn
import torchvision

import numpy as np
from model import *

DOWNLOAD_CIFAR = True

#prepare model
model_0 = CNN_4()
model_1 = CNN_4()
model_inter = CNN_4()

model_0 = model_0.cuda()
model_1 = model_1.cuda()
model_inter = model_inter.cuda()

model_0.load_state_dict(torch.load('./log/model_LR0.005.pt'))
model_1.load_state_dict(torch.load('./log/model_LR0.001.pt'))

#prepare data
train_data = torchvision.datasets.CIFAR10(root = './cifar/', train = True, download = DOWNLOAD_CIFAR)
test_data = torchvision.datasets.CIFAR10(root = './cifar/', train = False)

train_x = Variable(torch.from_numpy(train_data.train_data[:1000]).type(torch.FloatTensor).permute(0, 3, 1, 2)/255, volatile = True).cuda()
train_y = Variable(torch.LongTensor(train_data.train_labels[:1000]), volatile = True).cuda()

test_x = Variable(torch.from_numpy(test_data.test_data[:1000]).type(torch.FloatTensor).permute(0, 3, 1, 2)/255, volatile = True).cuda()
test_y = Variable(torch.LongTensor(test_data.test_labels[:1000]), volatile = True).cuda()

#help function
loss_func = nn.CrossEntropyLoss()
def inter_paras(model_0, model_1, model_inter, a):
	for (inter, m_0, m_1) in zip(model_inter.parameters(), model_0.parameters(), model_1.parameters()):
		inter.data = a*m_0.data + (1-a)*m_1.data

#interpolation
a_arr = np.linspace(-1, 2, 100)
accs = []
losses = []
test_accs = []
test_losses = []

for a in a_arr:
	inter_paras(model_0, model_1, model_inter, a)
	train_output = model_inter(train_x)
	train_loss = loss_func(train_output, train_y)
	train_pred = torch.squeeze(torch.max(train_output, 1)[1])
	train_acc = torch.sum(torch.eq(train_pred, train_y).type(torch.FloatTensor))/train_y.size()[0]

	test_output = model_inter(test_x)
	test_loss = loss_func(test_output, test_y)
	test_pred = torch.squeeze(torch.max(test_output, 1)[1])
	test_acc = torch.sum(torch.eq(test_pred, test_y).type(torch.FloatTensor))/test_y.size()[0]

	accs.append(train_acc.data[0])
	losses.append(train_loss.data[0])
	test_accs.append(test_acc.data[0])
	test_losses.append(test_loss.data[0])

print("a=%f:, loss: %f, acc: %f, test_loss: %f, test_acc: %f"%(a, train_loss.data[0], train_acc.data[0], test_loss.data[0], test_acc.data[0]))

np.save('log/accs.npy', accs)
np.save('log/losses.npy', losses)
np.save('log/test_accs.npy', test_accs)
np.save('log/test_losses.npy', test_losses)






