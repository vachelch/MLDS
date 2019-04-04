from my_models import *
import os
import numpy as np 
from utils import *



generator = Generator()
generator = generator.cuda()
generator.load_state_dict(torch.load('generator.pt'))

save_imgs(generator)




