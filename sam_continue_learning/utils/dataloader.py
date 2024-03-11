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
from utils.dataloader_sam import SamDataset
from utils.dataloader_ade20k import Ade20kDataset
from utils.dataloader_COCO import COCODataset
from utils.dataloader_VOC2012 import VOC2012Dataset
from utils.dataloader_cityscapes import CityScapesDataset
from utils.dataloader_gtea import GTEADataset
from utils.dataloader_LVIS import LVISDataset
from utils.dataloader_BBC038v1 import BBC03bv1Dataset
#### --------------------- dataloader online ---------------------####


def collate(batch):
    print(len(batch))
    
    return batch

def get_im_gt_name_dict(datasets, flag='valid', limit=-1):
    print("------------------------------", flag, "--------------------------------")
    name_im_gt_list = []

    for i in range(len(datasets)):        
        print("--->>>", flag, " dataset ",i,"/",len(datasets)," ",datasets[i]["name"],"<<<---")
        if "coco" in  datasets[i]["name"]:   
            name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                "root_dir": datasets[i]["im_dir"],
                "annotation_file": datasets[i]["annotation_file"]})
            print(datasets[i]["name"] + "continue")
            continue
        elif "LVIS" in  datasets[i]["name"]:   
            name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                "root_dir": datasets[i]["im_dir"],
                "annotation_file": datasets[i]["annotation_file"]})
            print(datasets[i]["name"] + "continue")
            continue
    
        
        tmp_im_list, tmp_gt_list = [], []
        for root, dirs, files in os.walk(datasets[i]["im_dir"]): 
            tmp_im_list.extend(glob(root+os.sep+'*'+datasets[i]["im_ext"]))

        #print(tmp_im_list)
        if 'DRAM' in datasets[i]["name"]:
            tmp_im_list = [x for x in tmp_im_list if 'train' not in x]
        
        #print(tmp_im_list)
        # raise NameError
        
        print(limit, flag,len(tmp_im_list))

        if flag=='train' and limit!=-1 and len(tmp_im_list)>limit:
            tmp_im_list=tmp_im_list[:limit]
        if "BBC038v1" in datasets[i]["name"]:   
            tmp_im_list = [x for x in tmp_im_list if 'masks' not in x]
            name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                    "im_path":tmp_im_list,
                                    "im_ext":datasets[i]["im_ext"]})
            
            print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
            print(len(name_im_gt_list))
            continue
                
        if(datasets[i]["gt_dir"]==""):
            tmp_gt_list = []
        else:
            tmp_gt_list = [x.replace(datasets[i]["im_ext"],datasets[i]["gt_ext"]).replace(datasets[i]["im_dir"],datasets[i]["gt_dir"]) for x in tmp_im_list]
            if 'DRAM' in datasets[i]["name"]:
                tmp_gt_list = [x.replace('test_images', 'test_targets_color') for x in tmp_gt_list]  
            
            # print(tmp_gt_list)
            # raise NameError
            tmp_gt_list = [x for x in tmp_gt_list if os.path.exists(x)]
            if 'DRAM' not in datasets[i]["name"]:
                tmp_im_list = [x for x in tmp_im_list if os.path.exists(x.replace(datasets[i]["im_ext"],datasets[i]["gt_ext"]).replace(datasets[i]["im_dir"],datasets[i]["gt_dir"]))]
        
        print('-im-',datasets[i]["name"],datasets[i]["im_dir"], ': ',len(tmp_im_list))
        print('-gt-', datasets[i]["name"],datasets[i]["gt_dir"], ': ',len(tmp_gt_list))


        name_im_gt_list.append({"dataset_name":datasets[i]["name"],
                                "im_path":tmp_im_list,
                                "gt_path":tmp_gt_list,
                                "im_ext":datasets[i]["im_ext"],
                                "gt_ext":datasets[i]["gt_ext"]})

    return name_im_gt_list

def create_dataloaders(name_im_gt_list, my_transforms=[], batch_size=1, batch_size_prompt=-1, batch_size_prompt_start=0, training=False, numworkers=-1):
    gos_dataloaders = []
    gos_datasets = []

    if(len(name_im_gt_list)==0):
        return gos_dataloaders, gos_datasets

    num_workers_ = 0
    if(batch_size>1):
        num_workers_ = 2
    if(batch_size>4):
        num_workers_ = 4
    if(batch_size>8):
        num_workers_ = 8
    if numworkers!=-1:  num_workers_ = numworkers
    
    if training:
        for i in range(len(name_im_gt_list)):
            if 'sam' in name_im_gt_list[i]['dataset_name']: gos_dataset = SamDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), batch_size_prompt=batch_size_prompt, batch_size_prompt_start=batch_size_prompt_start)
            else: gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms))
            gos_datasets.append(gos_dataset)

        gos_dataset = ConcatDataset(gos_datasets)
        sampler = DistributedSampler(gos_dataset)
        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler, batch_size, drop_last=True)
        dataloader = DataLoader(gos_dataset, batch_sampler=batch_sampler_train, num_workers=num_workers_)

        gos_dataloaders = dataloader
        gos_datasets = gos_dataset

    else:
        for i in range(len(name_im_gt_list)):  
            if 'sam' in name_im_gt_list[i]['dataset_name']: gos_dataset = SamDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True, batch_size_prompt=batch_size_prompt, batch_size_prompt_start=batch_size_prompt_start)
            #ADE dataloader 三通道 排除了[0 0 0]
            elif 'ADE' in name_im_gt_list[i]['dataset_name']: gos_dataset = Ade20kDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #cityscapes 单通道 什么也没排除
            elif 'cityscaps_val' in name_im_gt_list[i]['dataset_name']: gos_dataset = CityScapesDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #voc 三通道  排除了[0 0 0] [224 224 192]
            elif 'voc2012_val' in name_im_gt_list[i]['dataset_name']: gos_dataset = VOC2012Dataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #coco格式
            elif 'coco' in name_im_gt_list[i]['dataset_name']: gos_dataset = COCODataset(name_im_gt_list[i], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #DRAM 三通道 排除了[0 0 0]
            elif 'DRAM' in name_im_gt_list[i]['dataset_name']: gos_dataset = Ade20kDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #ppnls 三通道 排除了[0 0 0]
            elif 'ppdls' in name_im_gt_list[i]['dataset_name']: gos_dataset = Ade20kDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #gtea 单通道 排除了[0]
            elif 'gtea' in name_im_gt_list[i]['dataset_name']: gos_dataset = GTEADataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #ishape 单通道 排除了[0]
            elif 'ishape' in name_im_gt_list[i]['dataset_name']: gos_dataset = GTEADataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            #ishape 单通道 排除了[0]
            elif 'egohos' in name_im_gt_list[i]['dataset_name']: gos_dataset = GTEADataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            elif 'LVIS' in name_im_gt_list[i]['dataset_name']: gos_dataset = LVISDataset(name_im_gt_list[i], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            elif 'BBC038v1' in name_im_gt_list[i]['dataset_name']: gos_dataset = BBC03bv1Dataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)
            elif 'ZeroWaste' in name_im_gt_list[i]['dataset_name']: gos_dataset = GTEADataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution=True)

            
            else: gos_dataset = OnlineDataset([name_im_gt_list[i]], transform = transforms.Compose(my_transforms), eval_ori_resolution = True)
            sampler = DistributedSampler(gos_dataset, shuffle=False)
            dataloader = DataLoader(gos_dataset, batch_size, sampler=sampler, drop_last=False, num_workers=num_workers_)

            gos_dataloaders.append(dataloader)
            gos_datasets.append(gos_dataset)

    return gos_dataloaders, gos_datasets

class RandomHFlip(object):
    def __init__(self,prob=0.5):
        self.prob = prob
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        # random horizontal flip
        if random.random() >= self.prob:
            image = torch.flip(image,dims=[2])
            label = torch.flip(label,dims=[2])

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}

class Resize(object):
    def __init__(self,size=[320,320]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']        

        image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),self.size,mode='bilinear'),dim=0)
        label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),self.size,mode='bilinear'),dim=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}

class RandomCrop(object):
    def __init__(self,size=[288,288]):
        self.size = size
    def __call__(self,sample):
        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        h, w = image.shape[1:]
        new_h, new_w = self.size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[:,top:top+new_h,left:left+new_w]
        label = label[:,top:top+new_h,left:left+new_w]

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(self.size)}


class Normalize(object):
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,sample):

        imidx, image, label, shape =  sample['imidx'], sample['image'], sample['label'], sample['shape']
        image = normalize(image,self.mean,self.std)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':shape}



class LargeScaleJitter(object):
    """
        implementation of large scale jitter from copy_paste
        https://github.com/gaopengcuhk/Pretrained-Pix2Seq/blob/7d908d499212bfabd33aeaa838778a6bfb7b84cc/datasets/transforms.py 
    """

    def __init__(self, output_size=1024, aug_scale_min=0.1, aug_scale_max=2.0):
        self.desired_size = torch.tensor(output_size)
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max

    def pad_target(self, padding, target):
        target = target.copy()
        if "masks" in target:
            target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[1], 0, padding[0]))
        return target

    def __call__(self, sample):
        imidx, image, label, image_size =  sample['imidx'], sample['image'], sample['label'], sample['shape']

        #resize keep ratio
        out_desired_size = (self.desired_size * image_size / max(image_size)).round().int()

        random_scale = torch.rand(1) * (self.aug_scale_max - self.aug_scale_min) + self.aug_scale_min
        scaled_size = (random_scale * self.desired_size).round()

        scale = torch.minimum(scaled_size / image_size[0], scaled_size / image_size[1])
        scaled_size = (image_size * scale).round().long()
        
        scaled_image = torch.squeeze(F.interpolate(torch.unsqueeze(image,0),scaled_size.tolist(),mode='bilinear'),dim=0)
        scaled_label = torch.squeeze(F.interpolate(torch.unsqueeze(label,0),scaled_size.tolist(),mode='bilinear'),dim=0)

        # random crop
        crop_size = (min(self.desired_size, scaled_size[0]), min(self.desired_size, scaled_size[1]))

        margin_h = max(scaled_size[0] - crop_size[0], 0).item()
        margin_w = max(scaled_size[1] - crop_size[1], 0).item()
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0].item()
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1].item()

        scaled_image = scaled_image[:,crop_y1:crop_y2, crop_x1:crop_x2]
        scaled_label = scaled_label[:,crop_y1:crop_y2, crop_x1:crop_x2]

        # pad
        padding_h = max(self.desired_size - scaled_image.size(1), 0).item()
        padding_w = max(self.desired_size - scaled_image.size(2), 0).item()
        image = F.pad(scaled_image, [0,padding_w, 0,padding_h],value=128)
        label = F.pad(scaled_label, [0,padding_w, 0,padding_h],value=0)

        return {'imidx':imidx,'image':image, 'label':label, 'shape':torch.tensor(image.shape[-2:])}


class OnlineDataset(Dataset):
    def __init__(self, name_im_gt_list, transform=None, eval_ori_resolution=False):

        self.transform = transform
        self.dataset = {}
        ## combine different datasets into one
        dataset_names = []
        dt_name_list = [] # dataset name per image
        im_name_list = [] # image name
        im_path_list = [] # im path
        gt_path_list = [] # gt path
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

    def __len__(self):
        return len(self.dataset["im_path"])
    def __getitem__(self, idx):
        im_path = self.dataset["im_path"][idx]
        gt_path = self.dataset["gt_path"][idx]
        
        if 'sam' not in self.dataset["gt_path"][idx]:
            im = io.imread(im_path)
            gt = io.imread(gt_path)
        else:
            #print(im_path)
            all = io.imread(im_path)
            # print(all.shape)
            # print(type(all))
            im, gt = all[:,:512,:], all[:,522:1034,:]
            # import cv2
            # cv2.imwrite("demo_im.png", im)
            # cv2.imwrite("demo_gt.png", gt)
            # cv2.imwrite("demo.png", all)
            # raise NameError
            
        if len(gt.shape) > 2:
            gt = gt[:, :, 0]
        if len(im.shape) < 3:
            im = im[:, :, np.newaxis]
        if im.shape[2] == 1:
            im = np.repeat(im, 3, axis=2)
        im = torch.tensor(im.copy(), dtype=torch.float32)
        im = torch.transpose(torch.transpose(im,1,2),0,1)
        gt = torch.unsqueeze(torch.tensor(gt, dtype=torch.float32),0)
        # print(torch.max(im), torch.min(im))
        # print(torch.max(gt), torch.min(gt))
        # raise NameError
        sample = {
        "imidx": torch.from_numpy(np.array(idx)),
        "image": im,
        "label": gt,
        "shape": torch.tensor(im.shape[-2:]),
        }
        
        if self.transform:
            sample = self.transform(sample)

        if self.eval_ori_resolution:
            # print(torch.max(im))
            # raise NameError
            sample["ori_label"] = gt.type(torch.uint8)  # NOTE for evaluation only. And no flip here
            sample['ori_im_path'] = self.dataset["im_path"][idx]
            sample['ori_gt_path'] = self.dataset["gt_path"][idx]

        return sample