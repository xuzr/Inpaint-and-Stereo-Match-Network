import torch
import torch.nn as nn
import functools
import torchvision.models as models
from model.ResNetBlock import ResnetBlock


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

class ImgDecodeNet(nn.Module):
    def __init__(self, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(ImgDecodeNet, self).__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.outerdecode = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 1 * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1),)
        self.toimg=nn.Sequential(
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            nn.Conv2d(ngf+3, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU())

        self.outerdecoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 1 * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1),)
        self.toimgr=nn.Sequential(
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            ResnetBlock(ngf+3,'zero',norm_layer),
            nn.Conv2d(ngf+3, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.LeakyReLU())
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, x, y, xmix_ngf, ymix_ngf):
        xout = self.outerdecode(xmix_ngf)
        yout = self.outerdecoder(ymix_ngf)
        ximg = self.toimg(torch.cat([x, xout], 1))
        yimg = self.toimgr(torch.cat([y, yout], 1))
        
        return ximg,yimg
