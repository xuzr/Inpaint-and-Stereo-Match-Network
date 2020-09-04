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

    for _ in range(3, 3+int(random.random()*maxmasks)):
        sx = int(random.random() * (w-1))
        sy = int(random.random() * (h-1))
        w_ = min(int(random.random() * maxsize+50), w-1)
        h_ = min(int(random.random() * maxsize+50), h - 1)
        mask[sy:sy+h_, sx:sx+w_] = np.zeros_like(mask[sy:sy+h_, sx:sx+w_])
        
    return mask

class IASMNDataset(data.Dataset):
    def __init__(self,expose_imgs=True,normal_imgs=True,mask=True,disp=True,random_mask=False,max_masks=5,max_size=20,mask_path=None):
        super(IASMNDataset, self).__init__()
        self.random_mask = random_mask
        self.normal_imgs = normal_imgs
        self.expose_imgs = expose_imgs
        self.mask = mask
        self.disp = disp
        self.max_masks = max_masks
        self.max_size = max_size
        self.mask_path = mask_path
        
        
        
    def __len__(self):
        return self.get_len()

    def __getitem__(self, idx):
        samples = {}
        if self.random_mask:
            samples['imglnoh'], samples['imgrnoh'] = self.get_normal_imgs(idx)
            samples['displ'] = self.get_disp(idx)
            h, w = samples['imglnoh'].shape[:2]
            if self.mask_path:
                samples['oemaskl']=io.imread(os.path.join(self.mask_path,'{:08d}maskl.png'.format(idx)))
                samples['oemaskr']=io.imread(os.path.join(self.mask_path,'{:08d}maskr.png'.format(idx)))
            else:
                samples['oemaskl'] = generate_random_mask(h, w, self.max_masks, self.max_size)
                samples['oemaskr'] = generate_random_mask(h, w, self.max_masks, self.max_size)
            samples['imgl'] = samples['imglnoh'].copy()
            samples['imgl'][np.where(samples['oemaskl'] == 0)] = [1.0, 1.0, 1.0]
            samples['imgr'] = samples['imgrnoh'].copy()
            samples['imgr'][np.where(samples['oemaskr'] == 0)] = [1.0, 1.0, 1.0]
            samples['displ'][np.where(samples['oemaskl']== 0)]=0.0
            
        else:
            if self.expose_imgs:
                samples['imgl'], samples['imgr'] = self.get_expose_imgs(idx)
            
            if self.normal_imgs:
                samples['imglnoh'], samples['imgrnoh'] = self.get_normal_imgs(idx)
                 
            if self.mask:
                samples['oemaskl'], samples['oemaskr'] = self.get_mask(idx)
            else:
                samples['oemaskl'] = torch.sum(abs(samples['imgl']-samples['imglnoh']),1)<0.3
                samples['oemaskr'] = torch.sum(abs(samples['imgr']-samples['imgrnoh']),1)<0.3
                
            if self.disp:
                samples['displ'] = self.get_disp(idx)
        
        for key, value in samples.items():
            if len(value.shape) == 3:
                samples[key] = torch.from_numpy(value.transpose(2, 0, 1))
            elif len(value.shape) == 2:
                samples[key] = torch.from_numpy(value[np.newaxis,:,:])
        
        return samples
            
            
    def generate_mask(self,path,h=384,w=512):
        len_ = self.__len__()
        for i in range(len_):
            maskl = generate_random_mask(h, w, self.max_masks, self.max_size)
            maskr = generate_random_mask(h, w, self.max_masks, self.max_size)
            io.imsave(os.path.join(path,'{:08d}maskl.png'.format(i)),maskl)
            io.imsave(os.path.join(path,'{:08d}maskr.png'.format(i)),maskr)


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
