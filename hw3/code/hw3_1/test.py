from my_models import *
import os
import numpy as np 
from utils imort *


if not os.path.isdir('../../samples'):
    os.mkdir('../../samples')

generator = Generator()
generator = generator.cuda()
generator.load_state_dict(torch.load('generator.pt'))

save_imgs(generator)




