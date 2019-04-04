import torch.nn as nn
#build model
class CNN_1(nn.Module):
	def __init__(self):
		super(CNN_1, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 32,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(128*4*4, 128),
			nn.ReLU()
			)
		self.out = nn.Linear(128, 10)

	def forward(self, x):
		x = self.conv11(x)
		x = self.conv12(x)
	
		x = self.conv21(x)
		x = self.conv22(x)
	
		x = self.conv31(x)
		x = self.conv32(x)
	
		x = x.view(x.size(0), -1)
		x = self.dense1(x)
		x = self.out(x)
		return x

#build model
class CNN_2(nn.Module):
	def __init__(self):
		super(CNN_2, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 32,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(64*8*8, 64),
			nn.ReLU()
			)
		self.dense2 = nn.Sequential(
			nn.Linear(64, 64),
			nn.ReLU()
			)
		self.out = nn.Linear(64, 10)

	def forward(self, x):
		x = self.conv11(x)
		x = self.conv12(x)
	
		x = self.conv21(x)
		x = self.conv22(x)
	
		x = x.view(x.size(0), -1)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.out(x)
		return x
    
    
#build model
class CNN_3(nn.Module):
	def __init__(self):
		super(CNN_3, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(3, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(2)
			)

		self.dense = nn.Sequential(
			nn.Linear(32*16*16, 40),
			nn.ReLU()
			)
		self.out = nn.Linear(40, 10)

	def forward(self, x):
		x = self.conv(x)
	
		x = x.view(x.size(0), -1)
		x = self.dense(x)
		x = self.out(x)
		return x

#build model
class DNN_1(nn.Module):
	def __init__(self):
		super(DNN_1, self).__init__()
		self.dense1 = nn.Sequential(
			nn.Linear(28*28, 512),
			nn.ReLU()
			)

		self.dense2 = nn.Sequential(
			nn.Linear(512, 256),
			nn.ReLU()
			)
		self.dense3 = nn.Sequential(
			nn.Linear(256, 256),
			nn.ReLU()
			)
		self.out = nn.Linear(256, 10)

	def forward(self, x):
		x = x.view(x.size(0), -1)
		x = self.dense1(x)
		x = self.dense2(x)
		x = self.dense3(x)
		x = self.out(x)
		return x