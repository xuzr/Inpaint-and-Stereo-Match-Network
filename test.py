import torch
from model import UnetGenerator, StereoUnetGenerator
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from dataloader import BlenderSceneDataset
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

writer = SummaryWriter()
transform = transforms.Compose(
    [
        transforms.Resize((384, 640)),
        transforms.ToTensor()
    ])


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

optimizer = optim.Adam(modelG.parameters(), lr=1e-4, betas=(0.9, 0.999))


train_loader = data.DataLoader(BlenderSceneDataset("E://code//Data//scene02//scene02//lightfield//sequence", "./split/BlenderScene/train_files.txt",5,transform), batch_size=8, shuffle=True, num_workers=0, drop_last=True)


def write_tensorboard(imgl, imgr, imglnoh, imgrnoh, imglfake, imgrfake,maskl,maskr, loss,step):
    writer.add_image("imgl",imgl,step,dataformats='NCHW')
    writer.add_image("imgr",imgr,step,dataformats='NCHW')
    writer.add_image("imglnoh",imglnoh,step,dataformats='NCHW')
    writer.add_image("imgrnoh",imgrnoh,step,dataformats='NCHW')
    writer.add_image("imglfake",imglfake,step,dataformats='NCHW')
    writer.add_image("imgrfake", imgrfake, step,dataformats='NCHW')
    writer.add_image("maskl", maskl, step,dataformats='NCHW')
    writer.add_image("maskr", maskr, step,dataformats='NCHW')
    writer.add_scalar('loss', loss, step)
    step=step+1

def train(imgl, imgr, imglnoh, imgrnoh,maskl,maskr,step):
    imgl = imgl.cuda()
    imgr = imgr.cuda()
    imglnoh = imglnoh.cuda()
    imgrnoh = imgrnoh.cuda()
    maskl = maskl.cuda().byte()
    maskr = maskr.cuda().byte()
    optimizer.zero_grad()
    imglfake, imgrfake = modelG(imgl, imgr)
    loss = F.mse_loss(imglfake, imglnoh) + F.mse_loss(imgrfake, imgrnoh) + F.smooth_l1_loss(imglfake[maskl], imglnoh[maskl]) + F.smooth_l1_loss(imgrfake[maskr], imglnoh[maskr])
    
    write_tensorboard(imgl,imgr,imglnoh,imgrnoh,imglfake,imgrfake,maskl,maskr,loss,step)

    loss.backward()
    optimizer.step()
    




if __name__ == "__main__":
    step =0
    for epoch in range(400):
        for batch_idx, (imgl, imgr, imglnoh, imgrnoh,maskl,maskr) in enumerate(train_loader):
            train(imgl, imgr, imglnoh, imgrnoh,maskl,maskr,step)
            step =step+1