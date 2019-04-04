import torch 
from torch.autograd import Variable

# a = Variable(torch.randn(2, 10), requires_grad = True)
# y = a*2

# y.backward(torch.ones(a.size()))
# print(a.grad.data)

import torch
from torch.autograd import Variable
w1 = Variable(torch.Tensor([1.0,2.0,3.0]),requires_grad=True)
w2 = 2
d = w1 + 2 
d.backward(torch.ones(w1.size()))
print (w1.grad.data)