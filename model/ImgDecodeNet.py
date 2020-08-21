import torch
import torch.nn as nn
import functools
import torchvision.models as models
from model.ResNetBlock import ResnetBlock


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
            nn.Sigmoid())

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
            nn.Tanh())

    def forward(self, x, y, xmix_ngf, ymix_ngf):
        xout = self.outerdecode(xmix_ngf)
        yout = self.outerdecoder(ymix_ngf)
        ximg = self.toimg(torch.cat([x, xout], 1))
        yimg = self.toimgr(torch.cat([y, yout], 1))
        
        return ximg,yimg
