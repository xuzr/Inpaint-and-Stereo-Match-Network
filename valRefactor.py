import torch
from model import IASMNet
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from dataloader import BlenderSceneDataset,BlenderDataset,SceneflowDataset
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import numpy as np
import torch.nn as nn


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
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

modelG = IASMNet(3, 3, 7,maxdisp=args.max_disp)
modelG.cuda()
init_weights(modelG,'kaiming')

test_loader = data.DataLoader(SceneflowDataset('/home/kb457/Desktop/Data/sceneflow', "./split/Sceneflow/test_files.txt",mask_path='/home/kb457/Desktop/Data/test_mask'), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# def write_tensorboard(imgl, imgr, imglnoh, imgrnoh, imglfake, imgrfake,maskl,maskr, loss,step):
def write_tensorboard(scales, imgs, fre):
    if step % fre == 0:
        for key, value in imgs.items():
            writer.add_image(key,value,step,dataformats='NCHW')

    for key, value in scales.items():
        writer.add_scalar('train/'+key, value, step)
        


def test_batch(samples):

    imgl = samples['imgl'].cuda()
    imgr = samples['imgr'].cuda()
    imglnoh = samples['imglnoh'].cuda()
    imgrnoh = samples['imgrnoh'].cuda()
    depthl = samples['displ'].cuda()
    
    with torch.no_grad():
        outputs = modelG(imgl, imgr)
    depthl_pred = outputs['depthl']

    mask = depthl < args.max_disp
    maskzero = depthl > 0
    mask *= maskzero
    
    mae = F.l1_loss(depthl_pred.unsqueeze(0)[mask],depthl[mask])
    vaildoemask = torch.sum(abs(imgl - imglnoh), 1) < 0.1
    vaildoemask = vaildoemask.unsqueeze(0)
    vaildmae = F.l1_loss(depthl_pred.unsqueeze(0)[vaildoemask],depthl[vaildoemask])
    print("test step: {:06d}  mae:{:2.6f}".format(step,mae.item()))
    return mae,vaildmae
    

def test():
    modelG.eval()
    maeSum=0.0
    vaildMaeSum=0.0
    num_samples=0
    for batch_idx, samples in enumerate(test_loader):
        # mae_,vaildmae_ =test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr)
        mae_, vaildmae_ = test_batch(samples)
        if torch.isnan(mae_) or torch.isnan(vaildmae_):
            continue
        maeSum+=mae_
        vaildMaeSum+=vaildmae_
        num_samples+=1
    modelG.train()
    return maeSum/num_samples,vaildMaeSum/num_samples


if __name__ == "__main__":
    step =0
    minMae=None
    minOEMae=None
    for epoch in range(21):
        pretrain_dict = torch.load(args.loadmodel+'/checkpoint_{:04d}.tar'.format(epoch))
        modelG.load_state_dict(pretrain_dict['state_dict'])
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
            