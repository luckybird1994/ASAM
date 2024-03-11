import torch
from collections import OrderedDict

path1 = '../../pretrained_checkpoint/sam_vit_b_01ec64.pth'
checkpoint = torch.load(path1)
path2 = '../../pretrained_checkpoint/sam_vit_b_01ec64_maskdecoder.pth'
target_dict = OrderedDict()

for k,v in checkpoint.items():
    if 'mask_decoder' in k:
        target_dict[k[13:]] = v

torch.save(target_dict,path2)

