# Copyright by HQ-SAM team
# All rights reserved.

## data loader
from __future__ import print_function, division

import numpy as np
import random
from copy import deepcopy
from skimage import io
import os
from glob import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import json
from pycocotools.coco import COCO

class COCODataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, batch_size_prompt=-1):
        
        self.root_dir = name_im_gt_list['root_dir']
        self.transform = transform
        self.coco = COCO(name_im_gt_list['annotation_file'])
        
        self.image_ids = list(self.coco.imgs.keys())
        # Filter out image_ids without any annotations
        self.image_ids = [image_id for image_id in self.image_ids if len(self.coco.getAnnIds(imgIds=image_id)) > 0]
                
        # print('-im-',name_im_gt_list["dataset_name"],self.root_dir, ': ',len(self.image_ids))
        # raise NameError
        self.eval_ori_resolution = eval_ori_resolution
        self.batch_size_prompt = batch_size_prompt
        
        #To DO: open all instance
        self.all_instance = batch_size_prompt    
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]

        image_path = os.path.join(self.root_dir, image_info['file_name'])
        im = cv2.imread(image_path)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
            
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        
        ann_ids = self.coco.getAnnIds(imgIds=image_id)
        anns = self.coco.loadAnns(ann_ids)
        gt = []

        for ann in anns:
            try:
                decoded_mask = self.coco.annToMask(ann)
                gt.append(decoded_mask)
            except:
                continue
            
        gt = torch.tensor(np.stack(gt)).to(torch.float32) * 255.0

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),  
            "image": im,   # 3 H W
            "label": gt,   # N H W
            "shape": torch.tensor(im.shape[-2:]),
            }
        
        if self.transform: 
            sample = self.transform(sample)
        
        if self.eval_ori_resolution:
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = image_info['file_name']
            sample['ori_gt_path'] = True
        
        return sample