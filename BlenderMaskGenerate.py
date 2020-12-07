from skimage import io,morphology
import os
import numpy as np

TXT_PATH = './split/scene2random/test_files.txt'
DATA_PATH = '/data/scene2rand/lightfield/sequence'

paths = [line.strip() for line in open(TXT_PATH).readlines()]

for path in paths:
    path = paths[49]
    imglp, imgrp, imglnohp, imgrnohp = path.split()
    idx = imglp[:6]
    imgl = io.imread(os.path.join(DATA_PATH, imglp))
    imglnoh = io.imread(os.path.join(DATA_PATH, imglnohp))
    imgr = io.imread(os.path.join(DATA_PATH, imgrp))
    imgrnoh = io.imread(os.path.join(DATA_PATH, imgrnohp))

    maskl = np.where(abs(np.array(imgl).sum(2).astype(np.int16) - np.array(imglnoh).astype(np.int16).sum(2)) > 90, 0.0, 1.0)
    maskr = np.where(abs(np.array(imgr).sum(2).astype(np.int16) - np.array(imgrnoh).astype(np.int16).sum(2)) > 90, 0.0, 1.0)
    
    # maskl_closing = morphology.binary_closing(maskl,morphology.disk(31))
    # maskr_closing = morphology.binary_closing(maskr, morphology.disk(31))
    maskl_dilation = morphology.binary_erosion(maskl,morphology.disk(11))
    # maskl_dilation = morphology.binary_dilation(maskl,morphology.disk(5))
    
    io.imsave('maskl.png', maskl)
    
    io.imsave('maskl_closing.png',maskl_dilation.astype(np.float))
    import pdb; pdb.set_trace()



