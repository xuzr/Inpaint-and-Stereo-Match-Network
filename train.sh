# python train.py --loadmode pretrain_ckpt/0716/checkpoint_1900.tar 
# python train.py --loadmode pretrain_ckpt/0727/checkpoint_0420.tar --learningrate 1e-4
# python trainNew.py      --learningrate 1e-3 \
#                         --datapath /data/highlight/lightfield/sequence \
#                         --reconstruct_loss True \
#                         --loadmodel pretrained_ckpt/0802t2/checkpoint_0600.tar


python trainRefactor.py      --learningrate 1e-3 \
                        --datapath /data/highlight/lightfield/sequence \
                        --reconstruct_loss False \
                        # --loadmodel pretrained_ckpt/0802t2/checkpoint_0600.tar
