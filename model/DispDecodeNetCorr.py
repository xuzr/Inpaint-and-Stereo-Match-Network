import torch
import torch.nn as nn
import functools
import torchvision.models as modelss
from model.submodule import *


class hourglass(nn.Module):
    def __init__(self, inplanes):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(convbn_3d(inplanes, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1)

        self.conv3 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=2, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(convbn_3d(inplanes*2, inplanes*2, kernel_size=3, stride=1, pad=1),
                                   nn.ReLU(inplace=True))

        self.conv5 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes*2, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes*2)) #+conv2

        self.conv6 = nn.Sequential(nn.ConvTranspose3d(inplanes*2, inplanes, kernel_size=3, padding=1, output_padding=1, stride=2,bias=False),
                                   nn.BatchNorm3d(inplanes)) #+x

    def forward(self, x ,presqu, postsqu):
        
        out  = self.conv1(x) #in:1/4 out:1/8
        pre  = self.conv2(out) #in:1/8 out:1/8
        if postsqu is not None:
           pre = F.relu(pre + postsqu, inplace=True)
        else:
           pre = F.relu(pre, inplace=True)

        out  = self.conv3(pre) #in:1/8 out:1/16
        out  = self.conv4(out) #in:1/16 out:1/16

        if presqu is not None:
           post = F.relu(self.conv5(out)+presqu, inplace=True) #in:1/16 out:1/8
        else:
           post = F.relu(self.conv5(out)+pre, inplace=True) 

        out  = self.conv6(post)  #in:1/8 out:1/4

        return out, pre, post

class DispDecodeNet(nn.Module):
    def __init__(self, maxdisp, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(DispDecodeNet, self).__init__()

        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.maxdisp = maxdisp
        self.upngf = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ngf *2  * 3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf* 1))

        self.upngfr = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ngf *2  * 3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf* 1))

        self.dres0 = nn.Sequential(convbn_3d(ngf, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True),
                                     convbn_3d(32, 32, 3, 1, 1),
                                     nn.ReLU(inplace=True))

        self.dres1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   convbn_3d(32, 32, 3, 1, 1)) 

        self.dres2 = hourglass(32)

        self.dres3 = hourglass(32)

        self.dres4 = hourglass(32)

        self.classif1 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif2 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))

        self.classif3 = nn.Sequential(convbn_3d(32, 32, 3, 1, 1),
                                      nn.ReLU(inplace=True),
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False))
                                      
    def forward(self, x, y, xmix_2ngf, ymix_2ngf):
        refimg_fea = self.upngf(xmix_2ngf)
        targetimg_fea = self.upngfr(ymix_2ngf)

        #disp pred
        #matching
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        cost = torch.zeros([refimg_fea.size()[0], refimg_fea.size()[1], self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]]).cuda()

        for i in range(self.maxdisp // 4):
            if i > 0 :
            #  cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
            #  cost[:, refimg_fea.size()[1] :, i, :, i:] = targetimg_fea[:, :, :, :-i]
             cost[:, : , i, :, i:] = torch.mul(refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i])
             
             
            else:
            #  cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
            #  cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
             cost[:, :, i, :,:]   = torch.mul(refimg_fea,targetimg_fea)
        cost = cost.contiguous()


        cost0 = self.dres0(cost)
        cost0 = self.dres1(cost0) + cost0

        out1, pre1, post1 = self.dres2(cost0, None, None) 
        out1 = out1+cost0

        out2, pre2, post2 = self.dres3(out1, pre1, post1) 
        out2 = out2+cost0

        out3, pre3, post3 = self.dres4(out2, pre1, post2) 
        out3 = out3+cost0

        cost1 = self.classif1(out1)
        cost2 = self.classif2(out2) + cost1
        cost3 = self.classif3(out3) + cost2

        outputs={}
        if self.training:
            cost1 = F.upsample(cost1, [self.maxdisp,x.size()[2],x.size()[3]], mode='trilinear')
            cost2 = F.upsample(cost2, [self.maxdisp,x.size()[2],x.size()[3]], mode='trilinear')

            cost1 = torch.squeeze(cost1,1)
            pred1 = F.softmax(cost1,dim=1)
            pred1 = disparityregression(self.maxdisp)(pred1)

            cost2 = torch.squeeze(cost2,1)
            pred2 = F.softmax(cost2,dim=1)
            pred2 = disparityregression(self.maxdisp)(pred2)
            outputs['depthl_2ngf']=pred1
            outputs['depthl_ngf']=pred2

        cost3 = F.upsample(cost3, [self.maxdisp,x.size()[2],x.size()[3]], mode='trilinear')
        cost3 = torch.squeeze(cost3,1)
        pred3 = F.softmax(cost3,dim=1)
        
	#For your information: This formulation 'softmax(c)' learned "similarity" 
	#while 'softmax(-c)' learned 'matching cost' as mentioned in the paper.
	#However, 'c' or '-c' do not affect the performance because feature-based cost volume provided flexibility.
        pred3 = disparityregression(self.maxdisp)(pred3)
        outputs['depthl'] = pred3
        
        return outputs
