import torch
from torch import nn
m = nn.Identity(54, unused_argument1=0.1, unused_argument2=False)
test_input = torch.randn(128, 20)
output = m(test_input)
print(output.size())