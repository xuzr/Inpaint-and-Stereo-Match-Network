from dataloader.IASMNLoader import IASMNDataset
# from IASMNLoader import IASMNDataset
import os
import numpy as np
from PIL import Image
from skimage import io


class TextureDataset(IASMNDataset):
    def __init__(self, datafloder,txtpath,transform,expose_imgs=True, normal_imgs=False, mask=False, disp=True, random_mask=False, max_masks=5, max_size=20, mask_path=None, trainning=True):
        super().__init__(expose_imgs=expose_imgs, normal_imgs=normal_imgs, mask=mask, disp=disp, random_mask=random_mask, max_masks=max_masks, max_size=max_size, mask_path=mask_path, trainning=trainning)
        self.paths = [line.strip() for line in open(txtpath).readlines()]
        self.datafloder = datafloder
        self.transform = transform
        self.trainning = trainning
        self.w = 512
        self.h = 256
        
    def get_len(self):
        return len(self.paths)
    
    def get_expose_imgs(self, idx):
        if self.trainning:
            imglp, imgrp, _, _, _, _ = self.paths[idx].split()
        else:
            imglp, imgrp, _, _ = self.paths[idx].split()

        
        self.imgl = Image.open(os.path.join(self.datafloder, imglp))
        self.imgr = Image.open(os.path.join(self.datafloder, imgrp))
        
        self.imgl = self.imgl.resize((512, self.h), Image.NEAREST)
        self.imgr = self.imgr.resize((512, self.h), Image.NEAREST)

        self.imgl = np.array(self.imgl).astype(np.float32) / 255.0
        self.imgr = np.array(self.imgr).astype(np.float32) / 255.0

        
        return self.imgl[:,:,np.newaxis].repeat(3,2), self.imgr[:,:,np.newaxis].repeat(3,2)
        
    def get_disp(self, idx):
        if self.trainning:
            _, _, displp, _, _, _ = self.paths[idx].split()
        else:
            _, _, displp, _ = self.paths[idx].split()

        displ=io.imread(os.path.join(self.datafloder, displp))
        displ_img = Image.fromarray(displ.copy()*512.0/(3*640)).resize((512,self.h),Image.NEAREST)

        return np.array(displ_img).astype(np.float32)
    def get_mask(self, idx):
        return None,None


if __name__ == "__main__":
    dataset = TextureDataset('./', '../split/texture/train_files.txt', None,trainning=True)
    
    disp = dataset.get_disp(0)
    print('done')