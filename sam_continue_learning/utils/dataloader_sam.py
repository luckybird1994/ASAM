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

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, utils
from torchvision.transforms.functional import normalize
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
import json
from pycocotools import mask

class SamDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False, batch_size_prompt=-1, batch_size_prompt_start=0):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
        gt_num_list = [] # gt path
        im_ext_list = [] # im ext
        gt_ext_list = [] # gt ext
        for i in range(0,len(name_im_gt_list)):
            dataset_names.append(name_im_gt_list[i]["dataset_name"])
            # dataset name repeated based on the number of images in this dataset
            dt_name_list.extend([name_im_gt_list[i]["dataset_name"] for x in name_im_gt_list[i]["im_path"]])
            im_name_list.extend([x.split(os.sep)[-1].split(name_im_gt_list[i]["im_ext"])[0] for x in name_im_gt_list[i]["im_path"]])
                
            im_path_list.extend(name_im_gt_list[i]["im_path"])
            gt_path_list.extend(name_im_gt_list[i]["gt_path"])
                
            im_ext_list.extend([name_im_gt_list[i]["im_ext"] for x in name_im_gt_list[i]["im_path"]])
            gt_ext_list.extend([name_im_gt_list[i]["gt_ext"] for x in name_im_gt_list[i]["gt_path"]])


        self.dataset["data_name"] = dt_name_list
        self.dataset["im_name"] = im_name_list
        self.dataset["im_path"] = im_path_list
        self.dataset["ori_im_path"] = deepcopy(im_path_list)
        self.dataset["gt_path"] = gt_path_list
        self.dataset["ori_gt_path"] = deepcopy(gt_path_list)
        self.dataset["im_ext"] = im_ext_list
        self.dataset["gt_ext"] = gt_ext_list

        self.eval_ori_resolution = eval_ori_resolution
        self.batch_size_prompt = batch_size_prompt
        self.batch_size_prompt_start = batch_size_prompt_start
        
        #To DO: open all instance
        self.all_instance = batch_size_prompt

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]    
        gt_num = len(os.listdir(gt_path))

        
        try:
            im = io.imread(im_path)
        except Exception as e:
            print(f"Error occurred: {e}")
            print(f"Problematic image path: {im_path}")
            
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
                
        gt = None

        for i in range(self.batch_size_prompt_start, self.batch_size_prompt_start+self.batch_size_prompt):
            tmp_gt = io.imread(os.path.join(gt_path,'segmentation_'+str(i%gt_num)+'.png'))
            if len(tmp_gt.shape) > 2:
                tmp_gt = tmp_gt[:, :, 0]
            tmp_gt = np.expand_dims(tmp_gt, axis=0)

            if i==self.batch_size_prompt_start: gt = tmp_gt
            else: gt = np.concatenate([gt, tmp_gt])
            
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.tensor(gt, dtype=torch.float32)

        sample = {
            "imidx": torch.from_numpy(np.array(idx)),  
            "image": im,   # 3 H W
            "label": gt,   # N H W
            "shape": torch.tensor(im.shape[-2:]),
        }

        if self.transform: 
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            sample["ori_label"] = sample['label']  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = im_path
            sample['ori_gt_path'] = gt_path
        return sample