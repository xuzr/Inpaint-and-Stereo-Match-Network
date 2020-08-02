import torch
from model import UnetGenerator, StereoUnetGenerator
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from dataloader import BlenderSceneDataset
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='IPASMNet')

parser.add_argument('--loadmodel', default= None,
                    help='load model')
args = parser.parse_args()


writer = SummaryWriter()
transform = transforms.Compose([transforms.Resize((384, 640)),transforms.ToTensor()])


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

modelG = StereoUnetGenerator(3, 3, 7)
modelG.cuda()
init_weights(modelG,'kaiming')

optimizer = optim.Adam(modelG.parameters(), lr=1e-2, betas=(0.9, 0.999))

if args.loadmodel:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    modelG.load_state_dict(pretrain_dict['state_dict'])

train_loader = data.DataLoader(BlenderSceneDataset("/home/vodake/Data/Overexposed/Overexposed/scene01No1/lightfield/sequence", "./split/overexposed/train_files.txt",20,transform), batch_size=1, shuffle=True, num_workers=2, drop_last=True)
test_loader = data.DataLoader(BlenderSceneDataset("/home/vodake/Data/Overexposed/Overexposed/scene01No1/lightfield/sequence", "./split/overexposed/test_files.txt",20,transform), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# def write_tensorboard(imgl, imgr, imglnoh, imgrnoh, imglfake, imgrfake,maskl,maskr, loss,step):
def write_tensorboard(imgl, imgr, imglnoh, imgrnoh,imglfake, imgrfake, depthl,depthr,depthl_pred,depthr_pred,loss,step):
    if step%100==0:
        writer.add_image("imgl",imgl,step,dataformats='NCHW')
        writer.add_image("imgr",imgr,step,dataformats='NCHW')
        writer.add_image("imglnoh",imglnoh,step,dataformats='NCHW')
        writer.add_image("imgrnoh",imgrnoh,step,dataformats='NCHW')
        writer.add_image("imglfake",imglfake,step,dataformats='NCHW')
        writer.add_image("imgrfake", imgrfake, step,dataformats='NCHW')

        writer.add_image("depthl", depthl/depthl.max(), step,dataformats='NCHW')
        # writer.add_image("depthr", depthr/depthr.max(), step,dataformats='NCHW')
        writer.add_image("depthl_pred", (depthl_pred/depthl_pred.max()).unsqueeze(0), step,dataformats='NCHW')
        # writer.add_image("depthr_pred", depthr_pred/depthl_pred.max(), step,dataformats='NCHW')
    # writer.add_image("maskl", maskl, step,dataformats='NCHW')
    # writer.add_image("maskr", maskr, step,dataformats='NCHW')
    writer.add_scalar('train/loss', loss, step)
    step=step+1

def rwarp2l(right, displ):
    b,c,h,w=left.size()
    y0l,x0l=np.mgrid[0:h,0:w]
    yl = np.expand_dims(y0l, 0)
    yl = np.expand_dims(yl, 0).repeat(b,0)
    xl = np.expand_dims(x0l, 0)
    xl = np.expand_dims(xl, 0).repeat(b,0)
    #print(x.shape,y.shape)
    gridl = np.concatenate((xl, yl), 1)

    gridl = torch.from_numpy(gridl).cuda().float()
    y_zerosl = torch.zeros(displ.size()).cuda()
    flol = torch.cat((displ, y_zerosl), 1).float()

    #trans right to left
    gridl = gridl - flol
    
    #convert pos to [-1,1]
    gridw = 2.0 * gridl[:, 0, :, :] / max(w - 1, 1) - 1.0
    gridh = 2.0 * gridl[:, 1,:,:] / max(h - 1, 1) - 1.0
    gridw = torch.unsqueeze(gridw,dim=1)
    gridh = torch.unsqueeze(gridh,dim=1)
    vgridl = torch.cat((gridw, gridh), dim=1)
    
    vgridl = vgridl.permute(0, 2, 3, 1) 
    Irwarp2l=nn.functional.grid_sample(right,vgridl)

    return Irwarp2l


# def train(imgl, imgr, imglnoh, imgrnoh,maskl=None,maskr,step):
def train(imgl, imgr, imglnoh, imgrnoh,depthl,depthr,step):
    imgl = imgl.cuda()
    imgr = imgr.cuda()
    imglnoh = imglnoh.cuda()
    imgrnoh = imgrnoh.cuda()
    depthl=depthl.cuda()*2000
    depthr=depthr.cuda()*2000
    # maskl = maskl.cuda().byte()
    # maskr = maskr.cuda().byte()
    optimizer.zero_grad()
    outputs = modelG(imgl, imgr)
    imglfake, imgrfake = outputs['xout'],outputs['yout']
    depthl_pred = outputs['depthl']

    depth_loss = F.smooth_l1_loss(depthl_pred,depthl,reduction='mean') \
            + 0.5*F.smooth_l1_loss(outputs['depthl_2ngf'],depthl,reduction='mean') \
            + 0.7*F.smooth_l1_loss(outputs['depthl_ngf'],depthl,reduction='mean') 
    img_loss = F.mse_loss(imglfake, imglnoh,reduction='mean') + F.mse_loss(imgrfake, imgrnoh,reduction='mean')
    if step>5000:
        loss = depth_loss+0.5*img_loss
    else:
        loss = img_loss

    mae = F.l1_loss(depthl_pred,depthl)
    per = (abs(depthl_pred-depthl)/depthl).mean()
    # write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,maskl,maskr,loss,step)
    depthr_pred=None
    write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,depthl,depthr,depthl_pred,depthr_pred,loss,step)
    writer.add_scalar('train/mae', mae, step)
    writer.add_scalar('train/depth_loss', depth_loss, step)
    writer.add_scalar('train/err_per', per, step)

    loss.backward()
    optimizer.step()
    print("step: {:06d}".format(step))


def test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr):
    imgl = imgl.cuda()
    imgr = imgr.cuda()
    imglnoh = imglnoh.cuda()
    imgrnoh = imgrnoh.cuda()
    depthl=depthl.cuda()
    depthr=depthr.cuda()
    with torch.no_grad():
        outputs = modelG(imgl, imgr)
    # depthl_pred,depthr_pred = outputs['depthl'],outputs['depthr']
    depthl_pred = outputs['depthl']
    mae = F.l1_loss(depthl_pred,depthl)

    print("test step: {:06d}  mae:{:2.6f}".format(step,mae.item()))
    return mae
    

def test():
    modelG.eval()
    maeSum=0.0
    samples=0
    for batch_idx, (imgl, imgr, imglnoh, imgrnoh,depthl,depthr) in enumerate(test_loader):
        maeSum+=test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr)
        samples+=1
    modelG.train()
    return maeSum/samples


if __name__ == "__main__":
    step =0
    minMae=None
    for epoch in range(4000):
        for batch_idx, (imgl, imgr, imglnoh, imgrnoh,depthl,depthr) in enumerate(train_loader):
            train(imgl, imgr, imglnoh, imgrnoh,depthl,depthr,step)
            step =step+1

        if epoch%10==0:
            mae = test()
            if not minMae:
                minMae=mae
            else:
                minMae=min(minMae,mae)
        writer.add_scalar('vaild/mae', mae, epoch)
        writer.add_scalar('vaild/min_mae', minMae, epoch)
            
                
            
        if epoch%20==0:
            torch.save(
                {
                    'state_dict': modelG.state_dict()
                },
                "./ckpt/checkpoint_{:04d}.tar".format(epoch)
            )