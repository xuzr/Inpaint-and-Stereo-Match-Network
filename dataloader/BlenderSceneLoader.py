import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io
from PIL import Image
import numpy as np
import os


class BlenderSceneDataset(data.Dataset):
    def __init__(self, datafloder, txtpath,transform):
        self.paths = [line.strip() for line in open(txtpath).readlines()]
        self.datafloder = datafloder
        self.transform = transform
      

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        imgl, imgr, imglnoh, imgrnoh = self.paths[idx].split()
        imgl = Image.open(os.path.join(self.datafloder,imgl))
        imgr = Image.open(os.path.join(self.datafloder, imgr))
        
        imglnoh = Image.open(os.path.join(self.datafloder,imglnoh))
        imgrnoh = Image.open(os.path.join(self.datafloder, imgrnoh))

        return self.transform(imgl),self.transform(imgr),self.transform(imglnoh),self.transform(imgrnoh)
        


        

if __name__ == "__main__":
    train_files = open('./split/BlenderScene/train_files.txt','w')
    test_files = open('./split/BlenderScene/test_files.txt', 'w')
    
    count = 0

    imgl = 'Scene_Cam000_frame{:03d}.png'
    imgr = 'Scene_Cam001_frame{:03d}.png'
    imglnoh = 'Scene_NoHighLights_Cam000_frame{:03d}.png'
    imgrnoh = 'Scene_NoHighLights_Cam001_frame{:03d}.png'
    for _, dirs, files in os.walk("E:/code/Data/scene02/scene02/lightfield/sequence"):
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
                
            

