import torch
# from model import UnetGenerator, StereoUnetGenerator
from model import IASMNet

from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from dataloader import TextureDataset
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import argparse
from skimage import io
import numpy as np
import cv2
from utils import *


parser = argparse.ArgumentParser(description='IPASMNet')

parser.add_argument('--loadmodel', default= None,required=True,
                    help='load model')
parser.add_argument('--result', default= None,required=True,
                    help='path to save result')
args = parser.parse_args()

modelG = IASMNet(3, 3, 7,maxdisp=192)
modelG.cuda()

if args.loadmodel:
    print('Load pretrained model')
    pretrain_dict = torch.load(args.loadmodel)
    modelG.load_state_dict(pretrain_dict['state_dict'])

transform = transforms.Compose([transforms.Resize((384, 512)),transforms.ToTensor()])

test_loader = data.DataLoader(TextureDataset('./', "./split/texture/test_files.txt" ,transform,trainning=False), batch_size=1, shuffle=False, num_workers=0, drop_last=True)


def test_batch(imgl, imgr,depthl,step):
    with torch.no_grad():
        outputs = modelG(imgl, imgr)
    depthl_pred = outputs['depthl']
    mask=(depthl<192)*(depthl>0)
    mae = F.l1_loss(depthl_pred.unsqueeze(0)[mask],depthl[mask])
    print("test step: {:06d}  mae:{:2.6f}   oemae:{:2.6f}".format(step,mae.item(),0))
    return mae,0,depthl_pred,outputs['xout'],outputs['yout']

def test():
    modelG.eval()
    maeSum=0.0
    OEmaeSum=0.0
    samples_cnt=0
    for batch_idx, samples in enumerate(test_loader):
        imgl = samples['imgl'].cuda()
        imgr = samples['imgr'].cuda()
        depthl = samples['displ'].cuda()

        mae_,oemae_,depthl_pred,imgl_pred,imgr_pred=test_batch(imgl, imgr,depthl,batch_idx)
        maeSum+=mae_
        OEmaeSum+=oemae_
        samples_cnt+=1
        imglNP = imgl.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgrNP = imgr.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgl_predNP = imgl_pred.cpu().detach().numpy().squeeze().transpose(1,2,0)
        imgr_predNP = imgr_pred.cpu().detach().numpy().squeeze().transpose(1,2,0)
        
        depthl_predNP = depthl_pred.cpu().detach().numpy()[0]
        depthlNP = depthl.cpu().detach().numpy()[0][0]
        io.imsave(args.result+"/{:04d}imgl.png".format(batch_idx),imglNP)
        io.imsave(args.result+"/{:04d}imgr.png".format(batch_idx),imgrNP)


        io.imsave(args.result+"/{:04d}imglpred.png".format(batch_idx),imgl_predNP)
        io.imsave(args.result+"/{:04d}imgrpred.png".format(batch_idx),imgr_predNP)

        # io.imsave(args.result+"/{:04d}depthlgt.png".format(batch_idx),depthlNP)
        # io.imsave(args.result+"/{:04d}depthrgt.png".format(batch_idx),depthrNP)

        # io.imsave(args.result+"/{:04d}depthlpred.png".format(batch_idx),depthl_predNP)
        # io.imsave(args.result+"/{:04d}depthrpred.png".format(batch_idx),depthr_predNP)
        mask=(depthl<192)*(depthl>0)
        #import pdb; pdb.set_trace()
        masknp=mask.squeeze().detach().cpu().numpy()
        masknp=masknp[:,:,np.newaxis].repeat(3,2)
        savePFM(args.result+"/{:04d}depthlgt.pfm".format(batch_idx),depthlNP)
        savePFM(args.result+"/{:04d}depthlpred.pfm".format(batch_idx),depthl_predNP)
        io.imsave(args.result+"/{:04d}depthlgt.png".format(batch_idx),cv2.applyColorMap(depthlNP.astype(np.uint8),cv2.COLORMAP_JET)*masknp)
        #io.imsave(args.result+"/{:04d}depthrgt.png".format(batch_idx),cv2.applyColorMap((depthrNP*255).astype(np.uint8),cv2.COLORMAP_JET))
        io.imsave(args.result+"/{:04d}depthlpred.png".format(batch_idx),cv2.applyColorMap((depthl_predNP).astype(np.uint8),cv2.COLORMAP_JET))
        #io.imsave(args.result+"/{:04d}depthrpred.png".format(batch_idx),cv2.applyColorMap((depthr_predNP*255).astype(np.uint8),cv2.COLORMAP_JET))
        io.imsave(args.result+"/{:04d}depthlerr.png".format(batch_idx),abs(depthl_predNP-depthlNP)*masknp[:,:,0])
        #io.imsave(args.result+"/{:04d}depthrerr.png".format(batch_idx),abs(depthr_predNP-depthrNP))

        

    return maeSum/samples_cnt, OEmaeSum/samples_cnt


if __name__ == "__main__":
    step=0
    mae,oemae=test()

    print("ave mae:{:2.6f},ave oemae:{:2.6f}".format(mae,oemae))
