import torch
import torch.nn as nn
from torch.nn import functional as F 

batch_size = 32
block_size = 8
max_iters = 3000
eval_iterval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iter = 200

#---------------
torch.manual_seed(1337)

