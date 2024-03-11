import torch
from collections import OrderedDict

url = "HuggingFace-Demo/Space/sam_vit_b_01ec64.pth"  # can also be a local path
checkpoint = torch.load(url)

#0:21 1:26
url2 = 'HuggingFace-Demo/Space/asam_vit_b_decoder.pth'
checkpoint2 = torch.load(url2)

url3 = 'HuggingFace-Demo/Space/asam_vit_b.pth'

new_state_dict = {}
for k,v in checkpoint.items():
    new_state_dict[k]=v
    if 'mask_decoder' in k:
        new_state_dict[k] = checkpoint2[k[13:]]
        print(k)

torch.save(new_state_dict, url3)

print(torch.sum(new_state_dict['mask_decoder.mask_tokens.weight']), torch.sum(checkpoint['mask_decoder.mask_tokens.weight']))