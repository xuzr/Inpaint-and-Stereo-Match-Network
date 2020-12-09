from model.MixNet import MixNet
from model.DispDecodeNet import DispDecodeNet
from model.ImgDecodeNet import ImgDecodeNet
from model.FeatureMixNet import FeatureMixNet
import torch
import torch.nn as nn

class IASMNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, maxdisp=192,ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(IASMNet, self).__init__()
        # self.mixEncode = MixNetEncode(input_nc, num_downs, ngf, norm_layer, use_dropout)
        # self.mixDecode = MixNetDecode(output_nc, ngf, norm_layer, use_dropout)
        self.mixNet = MixNet(input_nc,output_nc,num_downs, ngf, norm_layer, use_dropout)
        self.dispDecode = DispDecodeNet(maxdisp, ngf, norm_layer, use_dropout)
        self.imgDecode = ImgDecodeNet(output_nc, ngf, norm_layer, use_dropout)
        self.featMix = FeatureMixNet(ngf)
    def forward(self, x, y):
        xmix_ngf, ymix_ngf, xmix_2ngf, ymix_2ngf = self.mixNet(x, y)
        outputs = self.dispDecode(x, y, xmix_2ngf, ymix_2ngf)
        xmix_ngf_mix = self.featMix(xmix_ngf,ymix_ngf.detach(),outputs['depthl'].detach())
        # xmix_ngf_mix = xmix_ngf
        ximg, yimg = self.imgDecode(x, y, xmix_ngf_mix, ymix_ngf)
        outputs['xout'] = ximg
        outputs['yout'] = yimg
        
        return outputs
        
        
