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
from skimage import io
import numpy as np
import cv2


parser = argparse.ArgumentParser(description='IPASMNet')

parser.add_argument('--loadmodel', default= None,
                    help='load model')
parser.add_argument('--result', default= None,
                    help='path to save result')
args = parser.parse_args()

modelG = StereoUnetGenerator(3, 3, 7)
modelG.cuda()

if args.loadmodel:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    modelG.load_state_dict(pretrain_dict['state_dict'])

transform = transforms.Compose([transforms.Resize((384, 640)),transforms.ToTensor()])

test_loader = data.DataLoader(BlenderSceneDataset("/home/vodake/Data/t2output/scene1/sequence", "./split/OEScene2/test_files.txt",20,transform), batch_size=1, shuffle=False, num_workers=0, drop_last=True)


def test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr):
    with torch.no_grad():
        outputs = modelG(imgl, imgr)
    depthl_pred,depthr_pred = outputs['depthl'],outputs['depthr']
    mae = F.l1_loss(depthl_pred,depthl)

    print("test step: {:06d}  mae:{:2.6f}".format(step,mae.item()))
    return mae,depthl_pred,depthr_pred,outputs['xout'],outputs['yout']

def test():
    modelG.eval()
    maeSum=0.0
    samples=0
    for batch_idx, (imgl, imgr, imglnoh, imgrnoh,depthl,depthr) in enumerate(test_loader):
        imgl = imgl.cuda()
        imgr = imgr.cuda()
        imglnoh = imglnoh.cuda()
        imgrnoh = imgrnoh.cuda()
        depthl=depthl.cuda()
        depthr=depthr.cuda()
        mae_,depthl_pred,depthr_pred,imgl_pred,imgr_pred=test_batch(imgl, imgr, imglnoh, imgrnoh,depthl,depthr)
        maeSum+=mae_
        samples+=1
        imglNP = imgl.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgrNP = imgr.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imglnohNP = imglnoh.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgrnohNP = imgrnoh.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgl_predNP = imgl_pred.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgr_predNP = imgr_pred.cpu().detach().numpy().squeeze().transpose(1,2,0)
        
        depthRange=0.2
        depthl_predNP= (depthl_pred.cpu().detach().numpy().squeeze()-0.05)/depthRange
        depthr_predNP= (depthr_pred.cpu().detach().numpy().squeeze()-0.05)/depthRange
        depthlNP = (depthl.cpu().detach().numpy().squeeze()-0.05)/depthRange
        depthrNP = (depthr.cpu().detach().numpy().squeeze()-0.05)/depthRange


        io.imsave(args.result+"/{:04d}imgl.png".format(batch_idx),imglNP)
        io.imsave(args.result+"/{:04d}imgr.png".format(batch_idx),imgrNP)

        io.imsave(args.result+"/{:04d}imglgt.png".format(batch_idx),imglnohNP)
        io.imsave(args.result+"/{:04d}imgrgt.png".format(batch_idx),imgrnohNP)

        io.imsave(args.result+"/{:04d}imglpred.png".format(batch_idx),imgl_predNP)
        io.imsave(args.result+"/{:04d}imgrpred.png".format(batch_idx),imgr_predNP)

        # io.imsave(args.result+"/{:04d}depthlgt.png".format(batch_idx),depthlNP)
        # io.imsave(args.result+"/{:04d}depthrgt.png".format(batch_idx),depthrNP)

        # io.imsave(args.result+"/{:04d}depthlpred.png".format(batch_idx),depthl_predNP)
        # io.imsave(args.result+"/{:04d}depthrpred.png".format(batch_idx),depthr_predNP)

        io.imsave(args.result+"/{:04d}depthlgt.png".format(batch_idx),cv2.applyColorMap((depthlNP*255).astype(np.uint8),cv2.COLORMAP_JET))
        io.imsave(args.result+"/{:04d}depthrgt.png".format(batch_idx),cv2.applyColorMap((depthrNP*255).astype(np.uint8),cv2.COLORMAP_JET))

        io.imsave(args.result+"/{:04d}depthlpred.png".format(batch_idx),cv2.applyColorMap((depthl_predNP*255).astype(np.uint8),cv2.COLORMAP_JET))
        io.imsave(args.result+"/{:04d}depthrpred.png".format(batch_idx),cv2.applyColorMap((depthr_predNP*255).astype(np.uint8),cv2.COLORMAP_JET))

        io.imsave(args.result+"/{:04d}depthlerr.png".format(batch_idx),abs(depthl_predNP-depthlNP))
        io.imsave(args.result+"/{:04d}depthrerr.png".format(batch_idx),abs(depthr_predNP-depthrNP))

        

    return maeSum/samples


if __name__ == "__main__":
    step=0
    print("ave mae:{:2.6f}".format(test()))
