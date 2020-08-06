# python train.py --loadmode pretrain_ckpt/0716/checkpoint_1900.tar 
# python train.py --loadmode pretrain_ckpt/0727/checkpoint_0420.tar --learningrate 1e-4
#python trainNew.py      --learningrate 1e-4 \
#                        --datapath /home/vodake/Data/highlight/lightfield/sequence \
#                        --loadmodel pretrain_ckpt/0801/checkpoint_0080.tar
                      
# python trainNew.py      --learningrate 1e-2 \
#                         --datapath /home/vodake/Data/highlight/lightfield/sequence \
#                         --loadmodel pretrain_ckpt/0801/checkpoint_0080.tar

python trainRepair.py   --learningrate 1e-2 \
                        --datapath /home/vodake/Data/highlight/lightfield/sequence \
                        # --loadmodel pretrain_ckpt/0801/checkpoint_0080.tar