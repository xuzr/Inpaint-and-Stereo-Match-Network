import torch.nn as nn
from model.ResNetBlock import ResnetBlock
import functools
import torch
import numpy as np

def rwarp2l(right, displ):
    b,c,h,w=right.size()
    y0l,x0l=np.mgrid[0:h,0:w]
    yl = np.expand_dims(y0l, 0)
    yl = np.expand_dims(yl, 0).repeat(b,0)
    xl = np.expand_dims(x0l, 0)
    xl = np.expand_dims(xl, 0).repeat(b,0)
    #print(x.shape,y.shape)
    gridl = np.concatenate((xl, yl), 1)


    gridl = torch.from_numpy(gridl).cuda().float()
    y_zerosl = torch.zeros(displ.size()).cuda()
                 
    flol=torch.cat((displ,y_zerosl),1).float()

    #trans right to left
    gridl=gridl-flol


	#convert pos to [-1,1]
    gridw = 2.0 * gridl[:, 0, :, :] / max(w - 1, 1) - 1.0
    gridh = 2.0 * gridl[:, 1,:,:] / max(h - 1, 1) - 1.0
    gridw = torch.unsqueeze(gridw,dim=1)
    gridh = torch.unsqueeze(gridh,dim=1)
    vgridl = torch.cat((gridw,gridh),dim=1)

    vgridl = vgridl.permute(0, 2, 3, 1) 

    Irwarp2l=nn.functional.grid_sample(right,vgridl)

    return Irwarp2l

class FeatureMixNet(nn.Module):
    def __init__(self, ngf, norm_layer=nn.BatchNorm2d):
        super(FeatureMixNet, self).__init__()
        
        self.ngf = ngf
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        self.weight_pred = nn.Sequential(
            ResnetBlock(ngf*6,'zero',norm_layer),
            ResnetBlock(ngf*6,'zero',norm_layer),
            ResnetBlock(ngf*6,'zero',norm_layer),
            nn.Conv2d(ngf*6, 1, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Sigmoid())

    
    def forward(self, target, source_, disp):
        weight = self.weight_pred(torch.cat([target, source_], dim=1))
        disp = disp.unsqueeze(1)
        disp = disp[:,:,::2,::2]
        warpd = rwarp2l(source_, disp)
        
        return weight*target+(1-weight)*warpd
        