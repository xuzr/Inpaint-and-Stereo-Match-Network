# python test.py --loadmode pretrain_ckpt/0716/checkpoint_1900.tar --result result/0716oe
# python test.py --loadmode pretrain_ckpt/0727/checkpoint_0420.tar --result result/0727
# python test.py --loadmode pretrain_ckpt/0728/checkpoint_0380.tar --result result/0728
# python testNew.py --loadmode pretrain_ckpt/0731/checkpoint_0280.tar --result result/0731
# python testNew.py --loadmode pretrained_ckpt/0811Repair/checkpoint_0037.tar --result result/0811_0037
# python valRefactor.py --loadmode pretrained_ckpt/0910sceneflow

# python testNew.py --loadmodel pretrained_ckpt/1121添加无监督lossckpt/checkpoint_0123.tar --result result/1121添加无监督/

# python testTexture.py --loadmodel pretrained_ckpt/201201Texture无监督loss/checkpoint_0085.tar --result result/201201Texture无监督loss

python testNew.py --loadmodel pretrained_ckpt/201207Scene2rand无监督loss/checkpoint_0087.tar --result result/201207s2/
