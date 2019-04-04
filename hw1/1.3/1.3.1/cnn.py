import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision

import numpy as np
import argparse
import time, os

from utils import *
from model import *

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = int, default = 4)
args = parser.parse_args()

if not os.path.isdir('./log'):
	os.mkdir('./log')


#Hyper parameters
EPOCH = 1000
BATCH_SIZE = 128
LR = 0.001
DOWNLOAD_MNIST = True


#preprocessing data
train_data = torchvision.datasets.MNIST(
	root = './mnist/',
	train = True,
	transform = torchvision.transforms.ToTensor(),
	download = DOWNLOAD_MNIST
	)

rand_ratio = 1
rand_size = int(len(train_data.train_labels) * rand_ratio)

train_X = torch.unsqueeze(train_data.train_data, dim = 1).type(torch.FloatTensor)/255
train_Y = train_data.train_labels.numpy()
train_Y[:rand_size] = np.random.randint(0, 10, rand_size)
train_Y = torch.LongTensor(train_Y)

test_data = torchvision.datasets.MNIST(root = './mnist/', train = False)

test_x = Variable(torch.unsqueeze(test_data.test_data[:1000], dim = 1).type(torch.FloatTensor)/255 , volatile = True).cuda()
test_y = Variable(torch.LongTensor(test_data.test_labels[:1000]), volatile = True).cuda()


if args.model == 1:
	cnn = CNN_1()
elif args.model == 2:
	cnn = CNN_2()
elif args.model == 3:
	cnn = CNN_3()
elif args.model == 4:
	cnn = DNN_1()


cnn = cnn.cuda()
print(cnn)
print("parameter amount: ", count_parameters(cnn))

optimizer = torch.optim.Adam(cnn.parameters(), lr = LR)
loss_func = nn.CrossEntropyLoss()



losses = []
test_losses = []

accs = []
test_accs = []

#training 
for epoch in range(EPOCH):
	start = time.time()
	for step in range(len(train_Y)//BATCH_SIZE):
		x = train_X[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
		y = train_Y[step*BATCH_SIZE : (step+1)*BATCH_SIZE]
		train_x = Variable(x.cuda())
		train_y = Variable(y.cuda())

		output = cnn(train_x)
		loss = loss_func(output, train_y)

		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

	#loss
	test_y_output = cnn(test_x)
	test_loss = loss_func(test_y_output, test_y)

	#acc
	pred = torch.squeeze(torch.max(output, 1)[1])
	test_pred = torch.squeeze(torch.max(test_y_output, 1)[1])

	acc = torch.sum(torch.eq(pred, train_y).type(torch.FloatTensor)) / train_y.size()[0]
	test_acc = torch.sum(torch.eq(test_pred, test_y).type(torch.FloatTensor)) / test_y.size()[0]

	#save accuracy and loss
	losses.append(loss.data[0])
	test_losses.append(test_loss.data[0])

	accs.append(acc.data[0])
	test_accs.append(test_acc.data[0])

	print("epoch%d %ds:, loss: %f, acc: %f, test_loss: %f, test_acc: %f"%(epoch + 1, int(time.time()-start),loss.data[0], acc.data[0], test_loss.data[0], test_acc.data[0]))
	with open('log/log_' + str(args.model) + '.txt', 'a') as f:
		f.write("epoch%d %ds:, loss: %f, acc: %f, test_loss: %f, test_acc: %f\n"%(epoch + 1, int(time.time()-start), loss.data[0], acc.data[0], test_loss.data[0], test_acc.data[0]))

torch.save(cnn.state_dict(), 'log/model_'+ str(args.model) + '.pt')

np.save("log/epoch.npy", [i+1 for i in range(EPOCH)])
np.save("log/losses_" + str(args.model) + ".npy", losses)
np.save("log/test_losses_" + str(args.model) + ".npy", test_losses)
np.save("log/accs_" + str(args.model) + ".npy", accs)
np.save("log/test_accs_" + str(args.model) + ".npy", test_accs)





















