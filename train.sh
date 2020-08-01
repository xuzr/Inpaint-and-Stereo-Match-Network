# python train.py --loadmode pretrain_ckpt/0716/checkpoint_1900.tar 
# python train.py --loadmode pretrain_ckpt/0727/checkpoint_0420.tar --learningrate 1e-4
python trainNew.py      --learningrate 1e-3 \
                        --datapath /home/kb457/data/t2output/scene1/sequence \
                        # --loadmodel pretrain_ckpt/0729/checkpoint_0020.tar
