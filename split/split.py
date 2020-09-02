import os


def splitSceneflow(path):
    train_file= open('Sceneflow/train_files.txt','w')
    test_file = open('Sceneflow/test_files.txt', 'w')
    
    img_com = 'frames_cleanpass'
    disp_com = 'disparity'

    train = 'TRAIN'
    test = 'TEST'

    labels = ['A', 'B', 'C']

    for label in labels:
        for _, dirs, _ in os.walk(os.path.join(path, img_com, train, label)):
            if len(dirs) < 100:
                continue
            for dir_idx in dirs:
                for idx in range(6, 16):
                    imgl = os.path.join(img_com,train,label,dir_idx,'left','{:04d}.png'.format(idx))
                    imgr = os.path.join(img_com,train,label,dir_idx,'right','{:04d}.png'.format(idx))
                    displ = os.path.join(disp_com,train,label,dir_idx,'left','{:04d}.pfm'.format(idx))
                    dispr = os.path.join(disp_com,train,label,dir_idx,'right','{:04d}.pfm'.format(idx))
                    train_file.write('{} {} {} {}\n'.format(imgl, imgr, displ, dispr))
    
    for label in labels:
        for _, dirs, _ in os.walk(os.path.join(path, img_com, test, label)):
            if len(dirs) < 100:
                continue
            for dir_idx in dirs:
                for idx in range(6,16):
                    imgl = os.path.join(img_com,test,label,dir_idx,'left','{:04d}.png'.format(idx))
                    imgr = os.path.join(img_com,test,label,dir_idx,'right','{:04d}.png'.format(idx))
                    displ = os.path.join(disp_com,test,label,dir_idx,'left','{:04d}.pfm'.format(idx))
                    dispr = os.path.join(disp_com,test,label,dir_idx,'right','{:04d}.pfm'.format(idx))
                    test_file.write('{} {} {} {}\n'.format(imgl,imgr,displ,dispr))


if __name__ == "__main__":
    splitSceneflow('/home/kb457/Desktop/Data/sceneflow')
