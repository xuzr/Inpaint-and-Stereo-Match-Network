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

optimizer = optim.Adam(modelG.parameters(), lr=1e-3, betas=(0.9, 0.999))

if args.loadmodel:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    modelG.load_state_dict(pretrain_dict['state_dict'])

train_loader = data.DataLoader(BlenderSceneDataset("/home/vodake/Data/t2output/scene1/sequence", "./split/OEScene2/train_files.txt",20,transform,True), batch_size=2, shuffle=True, num_workers=2, drop_last=True)
test_loader = data.DataLoader(BlenderSceneDataset("/home/vodake/Data/t2output/scene1/sequence", "./split/OEScene2/test_files.txt",20,transform), batch_size=1, shuffle=False, num_workers=0, drop_last=True)

# def write_tensorboard(imgl, imgr, imglnoh, imgrnoh, imglfake, imgrfake,maskl,maskr, loss,step):
def write_tensorboard(imgl, imgr, imglnoh, imgrnoh,imglfake, imgrfake, depthl,depthr,depthl_pred,depthr_pred,maskl,maskr,loss,step):
    if step%50==0:
        writer.add_image("imgl",imgl,step,dataformats='NCHW')
        writer.add_image("imgr",imgr,step,dataformats='NCHW')
        writer.add_image("imglnoh",imglnoh,step,dataformats='NCHW')
        writer.add_image("imgrnoh",imgrnoh,step,dataformats='NCHW')
        writer.add_image("imglfake",imglfake,step,dataformats='NCHW')
        writer.add_image("imgrfake", imgrfake, step,dataformats='NCHW')

        writer.add_image("depthl", depthl/depthl.max(), step,dataformats='NCHW')
        writer.add_image("depthr", depthr/depthr.max(), step,dataformats='NCHW')
        writer.add_image("depthl_pred", depthl_pred/depthl_pred.max(), step,dataformats='NCHW')
        writer.add_image("depthr_pred", depthr_pred/depthl_pred.max(), step,dataformats='NCHW')
        writer.add_image("maskl", maskl*255, step,dataformats='NCHW')
        writer.add_image("maskr", maskr*255, step,dataformats='NCHW')
    writer.add_scalar('train/loss', loss, step)
    step=step+1

# def train(imgl, imgr, imglnoh, imgrnoh,maskl=None,maskr,step):
def train(imgl, imgr, imglnoh, imgrnoh,depthl,depthr,maskl,maskr,step):
    imgl = imgl.cuda()
    imgr = imgr.cuda()
    imglnoh = imglnoh.cuda()
    imgrnoh = imgrnoh.cuda()
    depthl=depthl.cuda()*2000
    depthr=depthr.cuda()*2000
    # maskl_ = (depthl<=2000).byte()
    # maskr_ = (depthr<=2000).byte()
    maskl = maskl.cuda().byte()
    maskr = maskr.cuda().byte()
    # maskl=maskl*maskl_
    # maskr=maskr*maskr_
    optimizer.zero_grad()
    outputs = modelG(imgl, imgr)

    imglfake, imgrfake = outputs['xout'],outputs['yout']
    depthl_pred,depthr_pred = outputs['depthl']*2000,outputs['depthr']*2000

    # loss = F.mse_loss(imglfake, imglnoh) + F.mse_loss(imgrfake, imgrnoh)+100*F.smooth_l1_loss(depthl_pred,depthl)+100*F.smooth_l1_loss(depthr_pred,depthr) #+ 10*F.smooth_l1_loss(imglfake[maskl], imglnoh[maskl]) + F.smooth_l1_loss(imgrfake[maskr], imglnoh[maskr])
    # depth_loss = F.smooth_l1_loss(depthl_pred,depthl,reduction='mean')+F.smooth_l1_loss(depthr_pred,depthr,reduction='mean') \
    #         + 0.5*(F.smooth_l1_loss(outputs['depthl_2ngf'],depthl,reduction='mean')+F.smooth_l1_loss(outputs['depthr_2ngf'],depthr,reduction='mean')) \
    #         + 0.7*(F.smooth_l1_loss(outputs['depthl_ngf'],depthl,reduction='mean')+F.smooth_l1_loss(outputs['depthr_ngf'],depthr,reduction='mean'))
    # img_loss = F.mse_loss(imglfake, imglnoh,reduction='mean') + F.mse_loss(imgrfake, imgrnoh,reduction='mean')

    depth_loss = F.smooth_l1_loss(depthl_pred[maskl],depthl[maskl],reduction='mean')+F.smooth_l1_loss(depthr_pred[maskr],depthr[maskr],reduction='mean') \
            + 0.5*(F.smooth_l1_loss(outputs['depthl_2ngf'][maskl],depthl[maskl],reduction='mean')+F.smooth_l1_loss(outputs['depthr_2ngf'][maskr],depthr[maskr],reduction='mean')) \
            + 0.7*(F.smooth_l1_loss(outputs['depthl_ngf'][maskl],depthl[maskl],reduction='mean')+F.smooth_l1_loss(outputs['depthr_ngf'][maskr],depthr[maskr],reduction='mean'))
    oemaskl = 1-maskl
    oemaskr = 1-maskr
    oemaskl=oemaskl.repeat(1,3,1,1)
    oemaskr=oemaskr.repeat(1,3,1,1)
    img_loss = F.mse_loss(imglfake, imglnoh,reduction='mean') + F.mse_loss(imgrfake, imgrnoh,reduction='mean') \
                +100*(F.mse_loss(imglfake[oemaskl], imglnoh[oemaskl],reduction='mean') + F.mse_loss(imgrfake[oemaskr], imgrnoh[oemaskr],reduction='mean'))
    loss = depth_loss+0.5*img_loss
    # loss = img_loss
    mae = F.l1_loss(depthl_pred,depthl)
    per = (abs(depthl_pred-depthl)/depthl).mean()
    # write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,maskl,maskr,loss,step)
    write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,depthl,depthr,depthl_pred,depthr_pred,maskl,maskr,loss,step)
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
    depthl_pred,depthr_pred = outputs['depthl'],outputs['depthr']
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
        for batch_idx, (imgl, imgr, imglnoh, imgrnoh,depthl,depthr,maskl,maskr) in enumerate(train_loader):
            train(imgl, imgr, imglnoh, imgrnoh,depthl,depthr,maskl,maskr,step)
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