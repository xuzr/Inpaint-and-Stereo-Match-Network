import torch
import torch.nn as nn
import functools
import torchvision.models as models
from model.submodule import *

class MixNet(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MixNet, self).__init__()
        self.mixEncode = MixNetEncode(input_nc, num_downs, ngf, norm_layer, use_dropout)
        self.mixDecode = MixNetDecode(output_nc, ngf, norm_layer, use_dropout)

    def forward(self, x, y):
        encode_features = self.mixEncode(x, y)
        return self.mixDecode(encode_features)

class MixNetEncode(nn.Module):
    def __init__(self, input_nc, num_downs, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MixNetEncode, self).__init__()
        self.ngf = ngf

        #construct inner unet structure
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

    def forward(self, x, y):

        # x encode
        feature_ngf = self.outerencoder(x)
        feature_2ngf = self.down2ngf(feature_ngf)
        feature_4ngf = self.down4ngf(feature_2ngf)
        feature_8ngf = self.down8ngf(feature_4ngf)
        decode_8ngf = self.inner_blocks(feature_8ngf)[:, self.ngf*8:]

        # y encode
        yfeature_ngf = self.outerencoder(y)
        yfeature_2ngf = self.down2ngf(yfeature_ngf)
        yfeature_4ngf = self.down4ngf(yfeature_2ngf)
        yfeature_8ngf = self.down8ngf(yfeature_4ngf)
        ydecode_8ngf = self.inner_blocks(yfeature_8ngf)[:, self.ngf * 8:]

        encode_features = {}
        encode_features['x_ngf'] = feature_ngf
        encode_features['x_2ngf'] = feature_2ngf
        encode_features['x_4ngf'] = feature_4ngf
        encode_features['x_8ngf'] = feature_8ngf
        encode_features['x_dec8ngf'] = decode_8ngf

        encode_features['y_ngf'] = yfeature_ngf
        encode_features['y_2ngf'] = yfeature_2ngf
        encode_features['y_4ngf'] = yfeature_4ngf
        encode_features['y_8ngf'] = yfeature_8ngf
        encode_features['y_dec8ngf'] = ydecode_8ngf

        return encode_features
        

class MixNetDecode(nn.Module):
    def __init__(self, output_nc, ngf=32, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(MixNetDecode, self).__init__()

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        #construct decode structure
        self.up4ngf = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 4))
        self.up2ngf = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 *3, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 2))
        self.upngf2img = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf *2  * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1))


        self.up4ngfr = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8 * 3, ngf * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 4))
        self.up2ngfr = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4 *3, ngf * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 2))
        self.upngf2imgr = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf *2  * 3, ngf, kernel_size=4, stride=2, padding=1, bias=use_bias),
            norm_layer(ngf* 1))
            
    def forward(self, encode_features):
        # mix feature
        xmix_8ngf = torch.cat([encode_features['x_8ngf'], encode_features['x_dec8ngf'], encode_features['y_dec8ngf'].detach()],1)
        ymix_8ngf = torch.cat([encode_features['y_8ngf'], encode_features['x_dec8ngf'].detach(), encode_features['y_dec8ngf']],1)
        xdecode_4ngf = self.up4ngf(xmix_8ngf)
        ydecode_4ngf = self.up4ngfr(ymix_8ngf)

        xmix_4ngf = torch.cat([encode_features['x_4ngf'], xdecode_4ngf, ydecode_4ngf.detach()],1)
        ymix_4ngf = torch.cat([encode_features['y_4ngf'], xdecode_4ngf.detach(), ydecode_4ngf],1)
        xdecode_2ngf = self.up2ngf(xmix_4ngf)
        ydecode_2ngf = self.up2ngfr(ymix_4ngf)

        xmix_2ngf = torch.cat([encode_features['x_2ngf'], xdecode_2ngf, ydecode_2ngf.detach()],1)
        ymix_2ngf = torch.cat([encode_features['y_2ngf'], xdecode_2ngf.detach(), ydecode_2ngf],1)


        xdecode_ngf = self.upngf2img(xmix_2ngf)
        ydecode_ngf = self.upngf2imgr(ymix_2ngf)
        xmix_ngf = torch.cat([encode_features['x_ngf'], xdecode_ngf, ydecode_ngf.detach()],1)
        ymix_ngf = torch.cat([encode_features['y_ngf'], xdecode_ngf.detach(), ydecode_ngf], 1)

        return xmix_ngf,ymix_ngf,xmix_2ngf,ymix_2ngf
        
        

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
