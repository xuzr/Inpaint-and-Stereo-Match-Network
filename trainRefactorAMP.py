import torch
from model import IASMNet
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from dataloader import BlenderSceneDataset,BlenderDataset,SceneflowDataset,KittiDataset,TextureDataset
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import torch.nn as nn
from InpaintingLoss import InpaintingLoss,VGG16FeatureExtractor

LAMBDA_DICT = {
    'valid': 1.0, 'hole': 6.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

parser = argparse.ArgumentParser(description='IPASMNet')

parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--learningrate', type=float, default= 1e-3,
                    help='load model')
parser.add_argument('--datapath', default= None,
                    help='load model')
parser.add_argument('--max_disp', default=192, type=int, help='max disp')

parser.add_argument('--reconstruct_loss', type=boolean_string, default=True,
                    help='use reconstruct loss if True')

parser.add_argument('--fre_img', type=int, default= 50,
                    help='fre of write img to tensorboard')                  

args = parser.parse_args()


writer = SummaryWriter()
transform = transforms.Compose([transforms.Resize((384, 512)),transforms.ToTensor()])


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

modelG = IASMNet(3, 3, 7,maxdisp=args.max_disp,ngf=32)
modelG.cuda()
init_weights(modelG,'kaiming')

optimizer = optim.Adam(modelG.parameters(), lr=args.learningrate, betas=(0.9, 0.999))
scaler =torch.cuda.amp.GradScaler(enabled=False)
inpainting_loss = InpaintingLoss(VGG16FeatureExtractor()).cuda()

# if args.loadmodel:
#     print('Load pretrained model')
#     pretrain_dict = torch.load(args.loadmodel)
#     modelG.load_state_dict(pretrain_dict['state_dict'])

if args.loadmodel:
    print('Load pretrained model')
    tmp_dict = torch.load(args.loadmodel)['state_dict']
    modelG_dict = modelG.state_dict()
    modelG_dict.update(tmp_dict)
    modelG.load_state_dict(modelG_dict)

# train_loader = data.DataLoader(BlenderDataset(args.datapath, "./split/scene2random/train_files.txt",20,transform,True), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
# test_loader = data.DataLoader(BlenderDataset('/data/highlight/lightfield/sequence', "./split/scene2random/test_files.txt", 20, transform), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# train_loader = data.DataLoader(BlenderDataset(args.datapath, "./split/OEScene2/train_files.txt",20,transform,True), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
# test_loader = data.DataLoader(BlenderDataset('/data/highlight/lightfield/sequence', "./split/OEScene2/test_files.txt", 20, transform), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# train_loader = data.DataLoader(SceneflowDataset('/home/kb457/Desktop/Data/sceneflow', "./split/Sceneflow/train_files.txt",mask_path='/home/kb457/Desktop/Data/trainOneMask'), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
# test_loader = data.DataLoader(SceneflowDataset('/home/kb457/Desktop/Data/sceneflow', "./split/Sceneflow/test_files.txt",mask_path='/home/kb457/Desktop/Data/testOneMask'), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

train_loader = data.DataLoader(KittiDataset('/home/kb457/Desktop/Data', "./split/kitti2015/train_files.txt",mask_path='/home/kb457/Desktop/Data/trainRegionMask',trainning=True), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
test_loader = data.DataLoader(KittiDataset('/home/kb457/Desktop/Data', "./split/kitti2015/test_files.txt",mask_path='/home/kb457/Desktop/Data/testRegionMask'), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# train_loader = data.DataLoader(TextureDataset(args.datapath, "./split/texture/train_files.txt",transform,trainning=True), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
# test_loader = data.DataLoader(TextureDataset(args.datapath, "./split/texture/test_files.txt" ,transform,trainning=False), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

REPAIR=True

# def write_tensorboard(imgl, imgr, imglnoh, imgrnoh, imglfake, imgrfake,maskl,maskr, loss,step):
def write_tensorboard(scales, imgs, fre):
    if step % fre == 0:
        for key, value in imgs.items():
            writer.add_image(key,value,step,dataformats='NCHW')

    for key, value in scales.items():
        writer.add_scalar('train/'+key, value, step)


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

ssim = SSIM()
ssim=ssim.cuda()

def compute_reprojection_loss(pred, target):
    """Computes reprojection loss between a batch of predicted and target images
    """
    abs_diff = torch.abs(target - pred)
    l1_loss = abs_diff.mean(1, True)

    # ploss = PLoss(pred,target)

    ssim_loss = ssim(pred, target).mean(1, True)
    reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
    # reprojection_loss = 0.1 * ssim_loss + 0.9 * l1_loss


    return reprojection_loss

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

    Irwarp2l=nn.functional.grid_sample(right.float(),vgridl)

    return Irwarp2l

# def train(imgl, imgr, imglnoh, imgrnoh,maskl=None,maskr,step):
# def train(imgl, imgr, imglnoh, imgrnoh,depthl,depthr,maskl,maskr,step):


def train(samples, step):
    imgl = samples['imgl'].cuda()
    imgr = samples['imgr'].cuda()
    depthl=samples['displ'].cuda()

    if REPAIR:
        imglnoh = samples['imglnoh'].cuda()
        imgrnoh = samples['imgrnoh'].cuda()
        maskl = samples['oemaskl'].cuda().bool().detach()
        maskr = samples['oemaskr'].cuda().bool().detach()

    optimizer.zero_grad()
    with torch.cuda.amp.autocast(enabled=False):
        outputs = modelG(imgl, imgr)
        imglfake, imgrfake = outputs['xout'], outputs['yout']
        depthl_pred = outputs['depthl'].unsqueeze(1)
        mask = depthl < args.max_disp
        maskzero = depthl > 0
        mask *= maskzero
        
        
        # oemask = torch.sum(abs(imgl-imglnoh),1)<0.3
        # oemask=oemask.unsqueeze(0)
        # writer.add_image('oemask', oemask,global_step=step, dataformats='NCHW')
        if REPAIR:
            mask = mask*maskl
        mask=mask.bool().detach()
        depth_loss = F.smooth_l1_loss(depthl_pred[mask],depthl[mask],reduction='mean') \
                + 0.5*(F.smooth_l1_loss(outputs['depthl_2ngf'].unsqueeze(1)[mask],depthl[mask],reduction='mean')) \
                + 0.7*(F.smooth_l1_loss(outputs['depthl_ngf'].unsqueeze(1)[mask],depthl[mask],reduction='mean'))
        if REPAIR:
            oemaskl = ~maskl
            oemaskr = ~maskr
            oemaskl=oemaskl.repeat(1,3,1,1)
            oemaskr=oemaskr.repeat(1,3,1,1)
        # img_loss = F.mse_loss(imglfake, imglnoh, reduction='mean') + F.mse_loss(imgrfake, imgrnoh, reduction='mean')
        
        # if not oemaskl.sum() == 0:
        #     img_loss += F.mse_loss(imglfake[oemaskl], imglnoh[oemaskl], reduction='mean')
        # if not oemaskr.sum() == 0:
        #     img_loss += F.mse_loss(imgrfake[oemaskr], imgrnoh[oemaskr], reduction='mean')
            img_loss=0
            imgl_loss = inpainting_loss(imgl, maskl, imglfake, imglnoh)
        # imgr_loss = inpainting_loss(imgr,maskr,imgrfake,imgrnoh)

            for key, coef in LAMBDA_DICT.items():
                # img_loss +=(coef*(imgl_loss[key]+imgr_loss[key]))
                img_loss +=(coef*(imgl_loss[key]))
            # value = coef * loss_dict[key]
            # loss += value

        Irwarp2l = rwarp2l(imgrfake, depthl_pred)
        recon_loss = compute_reprojection_loss(imglfake, Irwarp2l).mean()
        if REPAIR:
            loss = depth_loss + img_loss + 0.5 * recon_loss
        else:
            loss = depth_loss + 0.5 * recon_loss
            

        
        
        # #loss = img_loss
        mask = depthl < args.max_disp
        maskzero = depthl > 0
        mask *=maskzero
        mae = F.l1_loss(depthl_pred[mask],depthl[mask])
        per = (abs(depthl_pred[mask] - depthl[mask]) / depthl[mask]).mean()

    # if args.reconstruct_loss:
    #     IFrwarp2l = rwarp2l(imgrfake, depthl_pred)
    #     reconmask = 1-oemask
    #     reconstructLoss = F.mse_loss(IFrwarp2l[reconmask*mask.repeat(1,3,1,1)], imglfake[reconmask*mask.repeat(1,3,1,1)])
    #     loss += reconstructLoss
    #     writer.add_scalar('train/rcloss', reconstructLoss, step)

    #     if step%50==0:
    #         writer.add_image("imglrecon", IFrwarp2l, step, dataformats='NCHW')
            
      
    # torch.cuda.empty_cache()
    # write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,maskl,maskr,loss,step)
    imgs = {}
    scales = {}
    imgs['imgl']=imgl
    imgs['imgr']=imgr

    imgs['imglfake']=imglfake
    imgs['imgrfake']=imgrfake
    imgs['depthl']=depthl/depthl.max()
    imgs['depthl_pred']=depthl_pred/depthl_pred.max()
    imgs['Irwarp2l']=Irwarp2l
    scales['loss']=loss
    scales['mae']=mae
    scales['depth_loss']=depth_loss
    scales['err_per'] = per
    if REPAIR:
        imgs['imglnoh']=imglnoh
        imgs['imgrnoh'] = imgrnoh
        imgs['maskl'] = maskl.float()
        
    write_tensorboard(scales,imgs,args.fre_img)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    # loss.backward()
    # optimizer.step()
    print("step: {:06d}".format(step))


def test_batch(samples):

    imgl = samples['imgl'].cuda()
    imgr = samples['imgr'].cuda()
    depthl = samples['displ'].cuda()

    if REPAIR:
        imglnoh = samples['imglnoh'].cuda()
        imgrnoh = samples['imgrnoh'].cuda()
        maskl = samples['oemaskl'].cuda().byte().detach()
        maskr = samples['oemaskr'].cuda().byte().detach()
    
    with torch.no_grad():
        outputs = modelG(imgl, imgr)
    depthl_pred = outputs['depthl']
    mask = depthl < args.max_disp
    maskzero = depthl > 0
    mask *=maskzero
    mae = F.l1_loss(depthl_pred.unsqueeze(0)[mask], depthl[mask])
    if REPAIR:
        vaildoemask = torch.sum(abs(imgl - imglnoh), 1) < 0.1
        vaildoemask = vaildoemask.unsqueeze(0)
        vaildmae = F.l1_loss(depthl_pred.unsqueeze(0)[vaildoemask], depthl[vaildoemask])
    else:
        vaildmae=0
    print("test step: {:06d}  mae:{:2.6f}".format(step,mae.item()))
    return mae,vaildmae
    

def test():
    modelG.eval()
    maeSum=0.0
    vaildMaeSum=0.0
    num_samples=0
    for batch_idx, samples in enumerate(test_loader):
        # mae_,vaildmae_ =test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr)
        mae_,vaildmae_ = test_batch(samples)
        maeSum+=mae_
        vaildMaeSum+=vaildmae_
        num_samples+=1
    modelG.train()
    return maeSum/num_samples,vaildMaeSum/num_samples


if __name__ == "__main__":
    step =0
    minMae=None
    minOEMae=None
    for epoch in range(301):
        for batch_idx, samples in enumerate(train_loader):
            train(samples,step)
            step =step+1

        if epoch%1==0:
            mae,oemae = test()
            if not minMae:
                minMae=mae
                minOEMae=oemae
            else:
                minMae=min(minMae,mae)
                minOEMae=min(minOEMae,oemae)
        writer.add_scalar('vaild/mae', mae, epoch)
        writer.add_scalar('vaild/min_mae', minMae, epoch)
        writer.add_scalar('vaild/oemae', oemae, epoch)
        writer.add_scalar('vaild/min_oemae', minOEMae, epoch)
            
                
            
        if epoch % 1 == 0:
            print(epoch)
            torch.save(
                {
                    'state_dict': modelG.state_dict(),
                    'scaler': scaler.state_dict()
                },
                "./ckpt/checkpoint_{:04d}.tar".format(epoch)
            )
