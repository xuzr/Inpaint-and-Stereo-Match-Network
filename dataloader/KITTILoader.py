from dataloader.IASMNLoader import IASMNDataset
# from IASMNLoader import IASMNDataset
import os
from PIL import Image
from skimage import io
import numpy as np
import random


class KittiDataset(IASMNDataset):
    def __init__(self, datafloder, txtpath,max_masks=5,max_size=20,mask_path=None,trainning=False):
        super(KittiDataset, self).__init__(random_mask=True,max_masks=max_masks,max_size=max_size,mask_path=mask_path)
        self.paths = [line.rstrip() for line in open(txtpath)]
        self.datafloder = datafloder
        self.trainning= trainning
        

    def get_len(self):
        return len(self.paths)-1

    def get_normal_imgs(self, idx):
        imgLPath, imgRPath, dispLPath,_ = self.paths[idx].split(' ')
        self.imgl = Image.open(os.path.join(self.datafloder,imgLPath)).convert('RGB')
        self.imgr = Image.open(os.path.join(self.datafloder, imgRPath)).convert('RGB')
        self.displ = Image.open(os.path.join(self.datafloder, dispLPath))
        if self.trainning:
            w, h = self.imgl.size
            th, tw = 256, 512
 
            self.x1 = random.randint(0, w - tw)
            self.y1 = random.randint(0, h - th)

            self.imgl = self.imgl.crop((self.x1, self.y1, self.x1 + tw, self.y1 + th))
            self.imgr = self.imgr.crop((self.x1, self.y1, self.x1 + tw, self.y1 + th))

            self.displ = np.ascontiguousarray(self.displ, dtype=np.float32) / 256
            self.displ = self.displ[self.y1:self.y1 + th, self.x1:self.x1 + tw]
        

        else:
            w, h = self.imgl.size
            self.imgl = self.imgl.crop((w-1024, h-384, w, h))
            self.imgr = self.imgr.crop((w - 1024, h - 384, w, h))
            w1, h1 = self.imgl.size

            self.displ = self.displ.crop((w - 1024, h - 384, w, h))
            self.displ = np.ascontiguousarray(self.displ, dtype=np.float32) / 256
            # self.displ = self.displ[h - 368:h, w - 1232:w]
            
        return np.array(self.imgl).astype(np.float32)/255.0,np.array(self.imgr).astype(np.float32)/255.0

    def get_disp(self, idx):
        return self.displ

    def get_mask(self, idx):
        maskl = io.imread(os.path.join(self.mask_path, '{:08d}maskl.png'.format(idx)))
        maskl = Image.fromarray(maskl.copy()).resize((512, 256), Image.NEAREST)

        maskr = io.imread(os.path.join(self.mask_path, '{:08d}maskr.png'.format(idx)))
        maskr = Image.fromarray(maskr.copy()).resize((512, 256), Image.NEAREST)

        return np.array(maskl).astype(np.float32), np.array(maskr).astype(np.float32)

if __name__ == "__main__":
    dataset = KittiDataset('/home/kb457/Desktop/Data', '../split/kitti2015/train_files.txt', max_masks=10,trainning=True)
    dataset.generate_mask('/home/kb457/Desktop/Data/data_scene_flow/train_mask',h=256,w=512)
    # dataset.get_normal_imgs(0)