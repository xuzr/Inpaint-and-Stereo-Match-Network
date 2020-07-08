import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io,morphology
from PIL import Image
import numpy as np
import os
import random



def generate_random_mask(h, w, maxmasks=10, maxsize=10):
    mask = np.ones((h, w))

    for i in range(3, 3+int(random.random()*maxmasks)):
        sx = int(random.random() * (w-1))
        sy = int(random.random() * (h-1))
        w_ = min(int(random.random()*maxsize+50), w-1)
        h_ = min(int(random.random() * maxsize+50), h - 1)
        mask[sy:sy+h_, sx:sx+w_] = np.zeros_like(mask[sy:sy+h_, sx:sx+w_])
        
    return mask
        



class BlenderSceneDataset(data.Dataset):
    def __init__(self, datafloder, txtpath,mor_size,transform):
        self.paths = [line.strip() for line in open(txtpath).readlines()]
        self.datafloder = datafloder
        self.transform = transform
        self.mor_size = mor_size
      

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        imgl, imgr, imglnoh, imgrnoh = self.paths[idx].split()
        imgl = Image.open(os.path.join(self.datafloder,imgl))
        imgr = Image.open(os.path.join(self.datafloder, imgr))
        
        imglnoh = Image.open(os.path.join(self.datafloder,imglnoh))
        imgrnoh = Image.open(os.path.join(self.datafloder, imgrnoh))

        # imgl = io.imread(os.path.join(self.datafloder,imgl))
        # imgr = io.imread(os.path.join(self.datafloder, imgr))

        # maskl = generate_random_mask(imgl.shape[0],imgl.shape[1],maxsize=200)[:,:,np.newaxis].repeat(3,2).astype(np.uint8)
        # maskr = generate_random_mask(imgl.shape[0], imgl.shape[1], maxsize=200)[:,:,np.newaxis].repeat(3,2).astype(np.uint8)
        
        # imglmask = np.ones_like(imgl)*255
        # imgrmask = np.ones_like(imgr) * 255
        
        # imglmask[np.where(maskl==1)]=imgl[np.where(maskl==1)]
        # imgrmask[np.where(maskr==1)]=imgr[np.where(maskr==1)]

        # imglmask = imgl * maskl
        
        # imgrmask = imgr * maskr

        # maskl = 1 - maskl
        # maskr = 1 - maskr
        
        
        # if(self.mor_size):
        #     kernel = morphology.disk(self.mor_size)
        #     maskl = morphology.dilation(maskl[:,:,0], kernel)[:,:,np.newaxis].repeat(3,2).astype(np.uint8)*255
            
        #     maskr = morphology.dilation(maskr[:,:,0], kernel)[:,:,np.newaxis].repeat(3,2).astype(np.uint8)*255

            




        return self.transform(imgl),self.transform(imgr),self.transform(imglnoh),self.transform(imgrnoh)
        # return self.transform(Image.fromarray(imglmask)),self.transform(Image.fromarray(imgrmask)),self.transform(Image.fromarray(imgl)),self.transform(Image.fromarray(imgr)), self.transform(Image.fromarray(maskl)), self.transform(Image.fromarray(maskr))
        


        

if __name__ == "__main__":
    train_files = open('./split/overexposed/train_files.txt','w')
    test_files = open('./split/overexposed/test_files.txt', 'w')
    
    count = 0

    imgl = 'Scene_Cam000_frame{:03d}.png'
    imgr = 'Scene_Cam001_frame{:03d}.png'
    imglnoh = 'Scene_NoHighLights_Cam000_frame{:03d}.png'
    imgrnoh = 'Scene_NoHighLights_Cam001_frame{:03d}.png'
    for _, dirs, files in os.walk("/home/vodake/Data/Overexposed/Overexposed/scene01No1/lightfield/sequence"):
        for dir in dirs:
            if (count % 5 == 0):
                test_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
                                 ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
            else:
                train_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
                                  ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
            count = count + 1
    
    train_files.close()
    test_files.close()
                
            

