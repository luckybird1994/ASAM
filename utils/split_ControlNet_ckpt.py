from diffusers import  ControlNetModel
import torch
from collections import OrderedDict

url = "../ckpt/control_v11p_sd15_canny.pth"  # can also be a local path

checkpoint = torch.load(url)
print(len(checkpoint.keys()))

#0:21 1:26
url2 = '../ControlNet-main/lightning_logs/version_26/checkpoints/epoch=45-step=21481.ckpt'
checkpoint2 = torch.load(url2)["state_dict"]

new_state_dict = {}
for k,v in checkpoint2.items():
    if k in checkpoint.keys():
        new_state_dict[k] = v

print(len(new_state_dict.keys()))
torch.save(new_state_dict,"../ckpt/control_v11p_sd15_mask_sa_000001.pth")
