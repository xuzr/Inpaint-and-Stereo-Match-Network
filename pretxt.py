import random

train_files = open('./split/OEScene2/train_files.txt','a+')
test_files = open('./split/OEScene2/test_files.txt', 'a+')

train_files = open('./split/scene2random/train_files.txt','a+')
test_files = open('./split/scene2random/test_files.txt', 'a+')

count = 0

imgl = 'Scene_Cam000_frame{:03d}.png'
imgr = 'Scene_Cam001_frame{:03d}.png'
imglnoh = 'Scene_NoHighLights_Cam000_frame{:03d}.png'
imgrnoh = 'Scene_NoHighLights_Cam001_frame{:03d}.png'
depthl = 'gt_depth_highres_Cam000.pfm'
depthr = 'gt_depth_highres_Cam001.pfm'
displ = 'gt_disp_highres_Cam000.pfm'
dispr = 'gt_disp_highres_Cam001.pfm'

# for _, dirs, files in os.walk("/home/vodake/Data/t2output/scene1/sequence"):
#     for dir in dirs:
#         if (count % 5 == 0):
#             test_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
#                              ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
#         else:
#             train_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
#                               ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
#         count = count + 1
idxs = [num for num in range(0, 301)]
random.shuffle(idxs)
print(idxs)
for idx in range(0, 301):
    dir = '{:06d}'.format(idxs[idx])
    if (count % 5 == 0):
        test_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
                            ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
    else:
        train_files.write(dir+'/'+imgl.format(int(dir)) + ' ' + dir+'/'+imgr.format(int(dir)) +
                            ' ' + dir+'/'+imglnoh.format(int(dir)) + ' ' + dir+'/'+imgrnoh.format(int(dir)) + '\n')
    count = count + 1


train_files.close()
test_files.close()