import torch 
from torch.autograd import Variable
import torch.nn as nn
import torchvision

import numpy as np
from model import *

DOWNLOAD_CIFAR = True

#prepare model
model_2 = CNN_4()
model_3 = CNN_4()
model_4 = CNN_4()
model_5 = CNN_4()
model_6 = CNN_4()

model_2 = model_2.cuda()
model_3 = model_3.cuda()
model_4 = model_4.cuda()
model_5 = model_5.cuda()
model_6 = model_6.cuda()

model_2.load_state_dict(torch.load('./log/model_LR0.005.pt'))
model_3.load_state_dict(torch.load('./log/model_LR0.004.pt'))
model_4.load_state_dict(torch.load('./log/model_LR0.003.pt'))
model_5.load_state_dict(torch.load('./log/model_LR0.002.pt'))
model_6.load_state_dict(torch.load('./log/model_LR0.001.pt'))

models = [model_2, model_3, model_4, model_5, model_6]

#prepare data
train_data = torchvision.datasets.CIFAR10(root = './cifar/', train = True, download = DOWNLOAD_CIFAR)
test_data = torchvision.datasets.CIFAR10(root = './cifar/', train = False)

train_x = Variable((torch.from_numpy(train_data.train_data[:1000]).type(torch.FloatTensor).permute(0, 3, 1, 2)/255).cuda(), requires_grad = True)
train_y = Variable(torch.LongTensor(train_data.train_labels[:1000])).cuda()

test_x = Variable((torch.from_numpy(test_data.test_data[:1000]).type(torch.FloatTensor).permute(0, 3, 1, 2)/255).cuda(), requires_grad = True)
test_y = Variable(torch.LongTensor(test_data.test_labels[:1000])).cuda()

#help function
loss_func = nn.CrossEntropyLoss()
def sensitivity(model, x, y):
	output = model(x)
	loss = loss_func(output, y)
	loss.backward()
	sensi = torch.sum(torch.norm(x.grad.data, 2, dim = 1).type(torch.FloatTensor))/y.size()[0]
	x.grad.data.zero_()
	return sensi



lrs = [0.005, 0.004, 0.003, 0.002, 0.001]
accs = []
losses = []
test_accs = []
test_losses = []
sensis = []
test_sensis = []

for model in models:
	sensi = sensitivity(model, train_x, train_y)
	test_sensi = sensitivity(model, test_x, test_y)
	train_output = model(train_x)
	train_loss = loss_func(train_output, train_y)
	train_pred = torch.squeeze(torch.max(train_output, 1)[1])
	train_acc = torch.sum(torch.eq(train_pred, train_y).type(torch.FloatTensor))/train_y.size()[0]

	test_output = model(test_x)
	test_loss = loss_func(test_output, test_y)
	test_pred = torch.squeeze(torch.max(test_output, 1)[1])
	test_acc = torch.sum(torch.eq(test_pred, test_y).type(torch.FloatTensor))/test_y.size()[0]

	accs.append(train_acc.data[0])
	losses.append(train_loss.data[0])
	test_accs.append(test_acc.data[0])
	test_losses.append(test_loss.data[0])
	sensis.append(sensi)
	test_sensis.append(test_sensi)

	print("sensi: %f, test_sensi: %f, loss: %f, acc: %f, test_loss: %f, test_acc: %f"%(sensi, test_sensi, train_loss.data[0], train_acc.data[0], test_loss.data[0], test_acc.data[0]))

np.save('log/lrs.npy', lrs)
np.save('log/accs.npy', accs)
np.save('log/losses.npy', losses)
np.save('log/test_accs.npy', test_accs)
np.save('log/test_losses.npy', test_losses)
np.save('log/sensi.npy', sensis)
np.save('log/test_sensi.npy', test_sensis)





