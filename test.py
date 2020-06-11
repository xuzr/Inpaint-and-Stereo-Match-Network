import torch
from model import UnetGenerator, StereoUnetGenerator

modelG = StereoUnetGenerator(3, 3, 7)
modelG.cuda()

x= torch.ones([1, 3, 384, 640], dtype=torch.float).cuda()
y= torch.ones([1, 3, 384, 640], dtype=torch.float).cuda()

x,y = modelG(x,y)

print(x.shape)
print(y.shape)