import numpy as np
import sys

def savePFM(path, image, scale=1):
        if image.dtype.name != 'float32':
            raise ValueError('Image dtype must be float32.')

        # color = self.color
        endian = image.dtype.byteorder
        if endian == '<' or endian == '=' and sys.byteorder == 'little':
            scale = -scale
        shape = image.shape
        # if self.compress:
        #     image = np.ascontiguousarray(compress(image, tolerance=0.5, parallel=False), dtype=np.uint8)
        with open(path, 'w') as file:
            file.write('Pf\n')
            # file.write('PF\n' if color else 'Pf\n')
            file.write('%d %d\n' % (shape[1], shape[0]))
            file.write('%f\n' % scale)

            values = np.flipud(np.asarray(image, dtype=np.float32))
            values.tofile(file)
        return image