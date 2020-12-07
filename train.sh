# python train.py --loadmode pretrain_ckpt/0716/checkpoint_1900.tar 
# python train.py --loadmode pretrain_ckpt/0727/checkpoint_0420.tar --learningrate 1e-4
# python trainNew.py      --learningrate 1e-3 \
#                         --datapath /data/highlight/lightfield/sequence \
#                         --reconstruct_loss True \
#                         --loadmodel pretrained_ckpt/0802t2/checkpoint_0600.tar


# python trainRefactor.py      --learningrate 1e-3 \
#                         --datapath /data/highlight/lightfield/sequence \
#                         --reconstruct_loss False \
#                         --loadmodel pretrained_ckpt/0910sceneflow/checkpoint_0020.tar
# #                         --loadmodel pretrained_ckpt/0802t2/checkpoint_0600.tar


# python trainRefactor.py      --learningrate 1e-3 \
#                         --reconstruct_loss False \
#                         --loadmodel pretrained_ckpt/0910sceneflow/checkpoint_0015.tar


# python trainTexture.py  --learningrate 1e-3 \
#                         --datapath ./ \
#                         --reconstruct_loss False \
#                         --loadmodel pretrained_ckpt/0910sceneflow/checkpoint_0020.tar

python trainRefactor.py      --learningrate 1e-3 \
                        --datapath /data/scene2rand/lightfield/sequence \
                        --reconstruct_loss False \
                        --loadmodel pretrained_ckpt/200910sceneflow/checkpoint_0020.tar