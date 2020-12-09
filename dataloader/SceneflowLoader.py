from dataloader.IASMNLoader import IASMNDataset
from dataloader.readPFM import readPFM
# from IASMNLoader import IASMNDataset
# from readPFM import readPFM
import os
from PIL import Image
import numpy as np
from skimage import io

class SceneflowDataset(IASMNDataset):
    def __init__(self, datafloder, txtpath,max_masks=5,max_size=20,mask_path=None):
        super(SceneflowDataset, self).__init__(random_mask=True,max_masks=max_masks,max_size=max_size,mask_path=mask_path)
        self.paths = [line.rstrip() for line in open(txtpath)]
        self.datafloder = datafloder
        

    def get_len(self):
        return len(self.paths)-1
        
    def get_normal_imgs(self, idx):
        imgLPath, imgRPath, _, _ = self.paths[idx].split(' ')
        self.imgl = Image.open(os.path.join(self.datafloder,imgLPath))
        self.imgr = Image.open(os.path.join(self.datafloder,imgRPath))
        self.imgl = self.imgl.resize((512, 256), Image.NEAREST)
        self.imgr = self.imgr.resize((512, 256), Image.NEAREST)

        return np.array(self.imgl).astype(np.float32)/255.0,np.array(self.imgr).astype(np.float32)/255.0

    def get_disp(self, idx):
        _, _, dispLPath, _ = self.paths[idx].split(' ')

        displ,_ = readPFM(os.path.join(self.datafloder,dispLPath))

        displ_img = Image.fromarray(displ.copy()*512.0/(960.0)).resize((512,256),Image.NEAREST)
        
        return np.array(displ_img).astype(np.float32)

    def get_mask(self, idx):

        maskl = io.imread(os.path.join(self.mask_path, '{:08d}maskl.png'.format(idx)))
        maskl = Image.fromarray(maskl.copy()).resize((512, 256), Image.NEAREST)

        maskr = io.imread(os.path.join(self.mask_path, '{:08d}maskr.png'.format(idx)))
        maskr = Image.fromarray(maskr.copy()).resize((512, 256), Image.NEAREST)

        return np.array(maskl).astype(np.float32), np.array(maskr).astype(np.float32)
        

if __name__ == "__main__":
    dataset = SceneflowDataset('/home/kb457/Desktop/Data/sceneflow', '../split/Sceneflow/test_files.txt', max_masks=10)
    dataset.generate_mask('/home/kb457/Desktop/Data/test_mask')
