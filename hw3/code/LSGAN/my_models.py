import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense = nn.Linear(100, 16*16*64)
        self.upsampling1 = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(64, 32, 3, 1, 1)
        self.upsampling2 = nn.Upsample(scale_factor=2)
        self.conv2 = nn.Conv2d(32, 16, 3, 1, 1)
        self.conv3 = nn.Conv2d(16, 3, 3, 1, 1)

    def forward(self, noise):
        x = (F.relu(self.dense(noise))).view(-1, 64, 16, 16)
        x = self.upsampling1(x)
        x = F.relu(self.conv1(x))
        x = self.upsampling2(x)
        x = F.relu(self.conv2(x))
        x = F.tanh(self.conv3(x))
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv11 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv12 = nn.Conv2d(16, 16, 3, 1, 1)
        self.MaxPool2d1 = nn.MaxPool2d(2)

        self.conv21 = nn.Conv2d(16, 32, 3, 1, 1)
        self.conv22 = nn.Conv2d(32, 32, 3, 1, 1)
        self.MaxPool2d2 = nn.MaxPool2d(2)

        self.conv31 = nn.Conv2d(32, 64, 3, 1, 1)
        self.conv32 = nn.Conv2d(64, 64, 3, 1, 1)
        self.MaxPool2d3 = nn.MaxPool2d(2)

        self.dense = nn.Linear(64*8*8, 64)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = F.relu(self.conv12(x))
        x = self.MaxPool2d1(x)

        x = F.relu(self.conv21(x))
        x = F.relu(self.conv22(x))
        x = self.MaxPool2d2(x)

        x = F.relu(self.conv31(x))
        x = F.relu(self.conv32(x))
        x = self.MaxPool2d3(x)
        x = x.view(x.size(0), -1)

        x = F.dropout(self.dense(x), 0.1, training=self.training)
        x = F.dropout(self.out(x), 0.1, training=self.training)

        return x

















