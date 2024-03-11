import torch
import cv2
import numpy as np

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks>128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks>128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

path_list = [
    'data/BIG/val/11083112904_5d97c7f8e1_o_sofa_gt.png', # example 4
    'data/CAMO/gts/camourflage_01118.png', # example 5
    'data/HRSOD-TE/gts/391544530_3232406e2d_o.png', #example 9
    '/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/sam_continue_learning/data/BIG/val/3686733971_b008837544_o_pottedplant_gt.png', #example2
    '/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/sam_continue_learning/data/HRSOD-TE/gts/30914871037_0b55479678_o.png', #example1
    '/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/sam_continue_learning/data/HRSOD-TE/gts/27449620621_186908e880_o.png' #example3
]

for path in path_list:
    mask = cv2.imread(path,0).astype(np.float32)
    mask = cv2.resize(mask, (1024,1024))
    mask = torch.tensor(mask).unsqueeze(0)
    #print(mask.shape)
    print(masks_to_boxes(mask))
    
    
    
    