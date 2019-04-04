import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import time

import numpy as np
import argparse, os

from utils import *
from model import *

torch.manual_seed(1)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type = int, default = 4)
parser.add_argument('--lr', type = float, default = 0.001)
args = parser.parse_args()

if not os.path.isdir('./log'):
	os.mkdir('./log')

#Hyper parameters
EPOCH = 30
BATCH_SIZE = 128
DOWNLOAD_CIFAR = True


# In[3]:


#preprocessing data
train_data = torchvision.datasets.CIFAR10(
	root = './cifar/',
	train = True,
	transform = torchvision.transforms.ToTensor(),
	download = DOWNLOAD_CIFAR
	)
test_data = torchvision.datasets.CIFAR10(root = './cifar/', train = False)
train_loader = Data.DataLoader(dataset = train_data, batch_size = BATCH_SIZE, shuffle = True)


# In[12]:


test_x = Variable(torch.from_numpy(test_data.test_data[:1000]).type(torch.FloatTensor).permute(0, 3, 1, 2)/255 , volatile = True).cuda()
test_y = Variable(torch.LongTensor(test_data.test_labels[:1000]), volatile = True).cuda()

if args.model == 0:
	cnn = CNN_0()
elif args.model == 1:
	cnn = CNN_1()
elif args.model == 2:
	cnn = CNN_2()
elif args.model == 3:
	cnn = CNN_3() 
elif args.model == 4:
	cnn = CNN_4()
elif args.model == 5:
	cnn = CNN_5()
elif args.model == 6:
	cnn = CNN_6()
elif args.model == 7:
	cnn = CNN_7() 
elif args.model == 8:
	cnn = CNN_8()
elif args.model == 9:
	cnn = CNN_9()
elif args.model == 10:
	cnn = CNN_10()


cnn = cnn.cuda()
print(cnn)
paras = count_parameters(cnn)
print("parameter amount: ", paras)


# In[15]:


optimizer = torch.optim.Adam(cnn.parameters(), lr = args.lr)
loss_func = nn.CrossEntropyLoss()


# In[60]:

losses = []
test_losses = []

accs = []
test_accs = []

#training 
for epoch in range(EPOCH):
	start = time.time()
	for step, (x, y) in enumerate(train_loader):
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
		if epoch == 0:
			f.write("parameters: %d\n"%(paras))
		f.write("epoch%d %ds:, loss: %f, acc: %f, test_loss: %f, test_acc: %f\n"%(epoch + 1, int(time.time()-start), loss.data[0], acc.data[0], test_loss.data[0], test_acc.data[0]))

torch.save(cnn.state_dict(), 'log/model_LR'+ str(args.lr) + '.pt')

np.save("log/losses_" + str(args.model) + ".npy", losses)
np.save("log/test_losses_" + str(args.model) + ".npy", test_losses)
np.save("log/accs_" + str(args.model) + ".npy", accs)
np.save("log/test_accs_" + str(args.model) + ".npy", test_accs)





















