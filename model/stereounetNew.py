import torch
import torch.nn as nn
import functools
import torchvision.models as models
from model.submodule import *


class Depth(nn.Module):
    def __init__(self,output_ch=1,input_chs={256,128,64},ngf=32):
        self.output_ch = output_ch
        self.input_chs = input_chs
        self.ngf=ngf

        for ch in self.input_chs:
            setattr(self,"")

class DepthDecode(nn.Module):
    def __inti__(self,input_ch,output_ch):
        super(DepthDecode,self).__init__()
        self.ngf=ngf
        self.input_ch=input_ch
        self.output_ch = output_ch

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


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

class StereoUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(StereoUnetGenerator, self).__init__()
        self.ngf = ngf
        # construct inner unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        self.inner_blocks = unet_block

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # construct encode structure
        self.outerencoder = nn.Sequential(nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias))
        self.down2ngf = nn.Sequential(nn.LeakyReLU(0.2,True),
                                        nn.Conv2d(ngf, ngf* 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                        norm_layer(ngf* 2))
        self.down4ngf = nn.Sequential(nn.LeakyReLU(0.2,True),
                                        nn.Conv2d(ngf*2, ngf* 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                        norm_layer(ngf* 4))
        self.down8ngf = nn.Sequential(nn.LeakyReLU(0.2,True),
                                        nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=use_bias),
                                        norm_layer(ngf* 8))

        #construct decode structure
        self.up4ngf = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 4))
        self.up2ngf = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 *3, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 2))
        self.upngf = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(ngf *2  * 3, ngf, kernel_size=3, stride=1, padding=1, bias=use_bias),
            norm_layer(ngf* 1))

        self.upngf2img = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf *2  * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1))
        self.outerdecode = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 1 * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1),
            ResnetBlock(ngf,'zero',norm_layer),
            ResnetBlock(ngf,'zero',norm_layer),
            ResnetBlock(ngf,'zero',norm_layer),
            ResnetBlock(ngf,'zero',norm_layer),
            nn.Conv2d(ngf, output_nc, kernel_size=3, stride=1, padding=1, bias=use_bias),
            nn.Tanh())

        self.maxdisp = 192
        self.dres0 = nn.Sequential(convbn_3d(64, 32, 3, 1, 1),
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
                                      nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1,bias=False))




    
    def forward(self, x, y):

        # x encode
        feature_ngf = self.outerencoder(x)
        feature_2ngf = self.down2ngf(feature_ngf)
        feature_4ngf = self.down4ngf(feature_2ngf)
        feature_8ngf = self.down8ngf(feature_4ngf)
        decode_8ngf = self.inner_blocks(feature_8ngf)[:,self.ngf*8:]

        # y encode
        yfeature_ngf = self.outerencoder(y)
        yfeature_2ngf = self.down2ngf(yfeature_ngf)
        yfeature_4ngf = self.down4ngf(yfeature_2ngf)
        yfeature_8ngf = self.down8ngf(yfeature_4ngf)
        ydecode_8ngf = self.inner_blocks(yfeature_8ngf)[:, self.ngf * 8:]

        # mix feature
        xmix_8ngf = torch.cat([feature_8ngf, decode_8ngf, ydecode_8ngf],1)
        ymix_8ngf = torch.cat([yfeature_8ngf, decode_8ngf, ydecode_8ngf],1)
        xdecode_4ngf = self.up4ngf(xmix_8ngf)
        ydecode_4ngf = self.up4ngf(ymix_8ngf)

        xmix_4ngf = torch.cat([feature_4ngf, xdecode_4ngf, ydecode_4ngf],1)
        ymix_4ngf = torch.cat([yfeature_4ngf, xdecode_4ngf, ydecode_4ngf],1)
        xdecode_2ngf = self.up2ngf(xmix_4ngf)
        ydecode_2ngf = self.up2ngf(ymix_4ngf)

        xmix_2ngf = torch.cat([feature_2ngf, xdecode_2ngf, ydecode_2ngf],1)
        ymix_2ngf = torch.cat([yfeature_2ngf, xdecode_2ngf, ydecode_2ngf],1)


        xdecode_ngf = self.upngf2img(xmix_2ngf)
        ydecode_ngf = self.upngf2img(ymix_2ngf)
        xmix_ngf = torch.cat([feature_ngf, xdecode_ngf, ydecode_ngf],1)
        ymix_ngf = torch.cat([yfeature_ngf, xdecode_ngf, ydecode_ngf],1)
        xout = self.outerdecode(xmix_ngf)
        yout = self.outerdecode(ymix_ngf)
        outputs={}
        outputs['xout']=xout
        outputs['yout']=yout

        refimg_fea = self.upngf(xmix_2ngf.detach())
        targetimg_fea = self.upngf(ymix_2ngf.detach())

        #disp pred
        #matching
        # cost = Variable(torch.FloatTensor(refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]).zero_()).cuda()
        cost = torch.zeros([refimg_fea.size()[0], refimg_fea.size()[1]*2, self.maxdisp//4,  refimg_fea.size()[2],  refimg_fea.size()[3]]).cuda()

        for i in range(self.maxdisp//4):
            if i > 0 :
             cost[:, :refimg_fea.size()[1], i, :,i:]   = refimg_fea[:,:,:,i:]
             cost[:, refimg_fea.size()[1]:, i, :,i:] = targetimg_fea[:,:,:,:-i]
            else:
             cost[:, :refimg_fea.size()[1], i, :,:]   = refimg_fea
             cost[:, refimg_fea.size()[1]:, i, :,:]   = targetimg_fea
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
        outputs['depthl']=pred3

        # xmix_ngf = torch.cat([feature_ngf, xdecode_ngf, ydecode_ngf],1)
        # ymix_ngf = torch.cat([yfeature_ngf, xdecode_ngf, ydecode_ngf],1)
        # xout = self.outerdecode(xmix_ngf)
        # yout = self.outerdecode(ymix_ngf)
        # outputs={}
        # outputs['x_4ngf']=xdecode_4ngf
        # outputs['x_2ngf']=xdecode_2ngf
        # outputs['x_ngf']=xdecode_ngf
        # outputs['x_out']=xout

        # outputs['y_4ngf']=ydecode_4ngf
        # outputs['y_2ngf']=ydecode_2ngf
        # outputs['y_ngf']=ydecode_ngf
        # outputs['y_out']=yout

        #decode depthl
        # depth_feature2ngfl = self.up2ngf_depthl(xdecode_4ngf)
        # depth_featurengfl = self.upngf_depthl(torch.cat([depth_feature2ngfl, xdecode_2ngf],1))
        # depthl = self.outerdecode_depthl(torch.cat([depth_featurengfl, xdecode_ngf],1))
        # #multi scale depthl
        # depth_2ngfl = self.trans2ngf2depthl(depth_feature2ngfl)
        # depth_ngfl = self.transngf2depthl(depth_featurengfl)

        #decode depthr
        # depth_feature2ngfr = self.up2ngf_depthr(ydecode_4ngf)
        # depth_featurengfr = self.upngf_depthr(torch.cat([depth_feature2ngfr, ydecode_2ngf],1))
        # depthr = self.outerdecode_depthr(torch.cat([depth_featurengfr, ydecode_ngf],1))
        # #multi scale depthl
        # depth_2ngfr = self.trans2ngf2depthr(depth_feature2ngfr)
        # depth_ngfr = self.transngf2depthr(depth_featurengfr)

        # outputs['depthl']=depthl
        # outputs['depthr']=depthr
        # outputs['xout']=xout
        # outputs['yout']=yout

        # depth_2ngfl=torch.nn.functional.upsample(depth_2ngfl,(depthl.shape[2],depthl.shape[3]),mode="bilinear")
        # depth_2ngfr=torch.nn.functional.upsample(depth_2ngfr,(depthr.shape[2],depthr.shape[3]),mode="bilinear")
        # outputs['depthl_2ngf']=depth_2ngfl
        # outputs['depthl_ngf']=depth_ngfl
        # outputs['depthr_2ngf']=depth_2ngfr
        # outputs['depthr_ngf']=depth_ngfr


        return outputs

        

class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)