import torch
from cnn import oneDcnn
x = torch.randn(1,1,512)
model = oneDcnn()
output = model(x)
print(output.shape)