import torch
import torch.nn as nn
import torch.utils.data as data
from skimage import io,morphology
from PIL import Image
import numpy as np
import os
import random
from dataloader.readPFM import readPFM

class IASMNDataset(data.Dataset):
    def __init__(self,random_mask=False,expose_imgs=True,normal_imgs=True,mask=True,disp=True):
        super(IASMNDataset, self).__init__()
        self.random_mask = random_mask
        self.normal_imgs = normal_imgs
        self.expose_imgs = expose_imgs
        self.mask = mask
        self.disp = disp
        
        
    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        samples = {}
        if self.random_mask:
            pass
        else:
            if self.expose_imgs:
                samples['imgl'], samples['imgr'] = self.get_expose_imgs(idx)
            
            if self.normal_imgs:
                samples['imglnoh'], samples['imgrnoh'] = self.get_normal_imgs(idx)
                 
            if self.mask:
                samples['oemaskl'], samples['oemaskr'] = self.get_mask(idx)
                
            if self.disp:
                samples['displ'] = self.get_disp(idx)
        
        for key, value in samples.items():
            if len(value.shape) == 3:
                samples[key] = torch.from_numpy(value.transpose(2, 0, 1))
            elif len(value.shape) == 2:
                samples[key] = torch.from_numpy(value[np.newaxis,:,:])
        
        return samples
            
            
        

    def get_len(self):
        raise NotImplementedError

    def get_normal_imgs(self, idx):
        raise NotImplementedError
        
    def get_expose_imgs(self, idx):
        raise NotImplementedError

    def get_disp(self, idx):
        raise NotImplementedError

    def get_mask(self, idx):
        raise NotImplementedError
