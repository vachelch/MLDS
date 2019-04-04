import torch.nn as nn
#build model
class CNN_0(nn.Module):
	def __init__(self):
		super(CNN_0, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 4,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(4, 4, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(4, 8, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(8, 8, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(8, 16, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(16, 16, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(16*4*4, 64),
			nn.ReLU()
			)
		self.out = nn.Linear(64, 10)

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
class CNN_1(nn.Module):
	def __init__(self):
		super(CNN_1, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 8,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(8, 8, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(8, 16, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(16, 16, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(32*4*4, 64),
			nn.ReLU()
			)
		self.out = nn.Linear(64, 10)

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
				out_channels = 16,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(16, 16, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(16, 32, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(32, 32, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(32, 64, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(64*4*4, 64),
			nn.ReLU()
			)
		self.out = nn.Linear(64, 10)

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
class CNN_3(nn.Module):
	def __init__(self):
		super(CNN_3, self).__init__()
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
			nn.Conv2d(64, 96, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(96, 96, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(96*4*4, 64),
			nn.ReLU()
			)
		self.out = nn.Linear(64, 10)

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
class CNN_4(nn.Module):
	def __init__(self):
		super(CNN_4, self).__init__()
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
class CNN_5(nn.Module):
	def __init__(self):
		super(CNN_5, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 48,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(48, 48, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(48, 96, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(96, 96, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(96, 128, 3, 1, 1),
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
class CNN_6(nn.Module):
	def __init__(self):
		super(CNN_6, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 64,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(128, 192, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(192, 192, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(192*4*4, 128),
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
class CNN_7(nn.Module):
	def __init__(self):
		super(CNN_7, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 64,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(64, 64, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(64, 128, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(256*4*4, 128),
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
class CNN_8(nn.Module):
	def __init__(self):
		super(CNN_8, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 128,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(128, 192, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(192, 192, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(192, 256, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(256*4*4, 128),
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
class CNN_9(nn.Module):
	def __init__(self):
		super(CNN_9, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 128,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(128, 128, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(128, 256, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(256, 512, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(512*4*4, 128),
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
class CNN_10(nn.Module):
	def __init__(self):
		super(CNN_10, self).__init__()
		self.conv11 = nn.Sequential(
			nn.Conv2d(
				in_channels = 3,
				out_channels = 256,
				kernel_size = 3,
				stride = 1,
				padding = 1
				),
			nn.ReLU()
			)
		self.conv12 = nn.Sequential(
			nn.Conv2d(256, 256, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv21 = nn.Sequential(
			nn.Conv2d(256, 384, 3, 1, 1),
			nn.ReLU()
			)
		self.conv22 = nn.Sequential(
			nn.Conv2d(384, 384, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.conv31 = nn.Sequential(
			nn.Conv2d(384, 512, 3, 1, 1),
			nn.ReLU()
			)
		self.conv32 = nn.Sequential(
			nn.Conv2d(512, 512, 3, 1, 1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size = 2)
			)

		self.dense1 = nn.Sequential(
			nn.Linear(512*4*4, 128),
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