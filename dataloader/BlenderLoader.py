from dataloader.IASMNLoader import IASMNDataset
import os
from PIL import Image
import numpy as np
from dataloader.readPFM import readPFM


class BlenderDataset(IASMNDataset):
    def __init__(self, datafloder, txtpath, mor_size, transform, ob=False, random_mask=False, expose_imgs=True, normal_imgs=True, mask=True, disp=True):
        super(BlenderDataset,self).__init__(random_mask,expose_imgs,normal_imgs,mask,disp)
        self.paths = [line.strip() for line in open(txtpath).readlines()]
        self.datafloder = datafloder
        self.transform = transform
        self.mor_size = mor_size
        self.ob = ob

    def get_len(self):
        return len(self.paths)

    def get_expose_imgs(self, idx):
        imglp, imgrp, _, _ = self.paths[idx].split()

        self.imgl = Image.open(os.path.join(self.datafloder,imglp))
        self.imgr = Image.open(os.path.join(self.datafloder, imgrp))

        self.imgl = self.imgl.resize((512, 384), Image.NEAREST)
        self.imgr = self.imgr.resize((512, 384), Image.NEAREST)
        
        return np.array(self.imgl).astype(np.float32)/255.0,np.array(self.imgr).astype(np.float32)/255.0

    def get_normal_imgs(self, idx):
        _, _, imglnohp, imgrnohp = self.paths[idx].split()

        self.imglnoh = Image.open(os.path.join(self.datafloder,imglnohp))
        self.imgrnoh = Image.open(os.path.join(self.datafloder, imgrnohp))

        self.imglnoh = self.imglnoh.resize((512, 384), Image.NEAREST)
        self.imgrnoh = self.imgrnoh.resize((512, 384), Image.NEAREST)
        
        return np.array(self.imglnoh).astype(np.float32)/255.0, np.array(self.imgrnoh).astype(np.float32)/255.0
        
    def get_mask(self, idx):
        maskl= np.where(abs(np.array(self.imgl).sum(2).astype(np.int16)-np.array(self.imglnoh).astype(np.int16).sum(2))>90,0.0,1.0)
        maskr = np.where(abs(np.array(self.imgr).sum(2).astype(np.int16) - np.array(self.imgrnoh).astype(np.int16).sum(2)) > 90, 0.0, 1.0)
        
        return maskl.astype(np.float32), maskr.astype(np.float32)
        
    def get_disp(self, idx):
        imgl, _, _, _ = self.paths[idx].split()
        file_idx = imgl[:6]

        displ_file = 'gt_disp_highres_Cam000.pfm'
        displ,_ = readPFM(os.path.join(self.datafloder,file_idx,displ_file))

        displ_img = Image.fromarray(displ.copy()*512.0/(3*640)).resize((512,384),Image.NEAREST)

        return np.array(displ_img).astype(np.float32)