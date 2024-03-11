import os
import cv2
import shutil
from pycocotools  import mask
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm

dir = '/data/tanglv/data/sa_000001/'
new_dir = '../sa_000001/'

files = os.listdir(dir)
Path(new_dir).mkdir(exist_ok=True, parents=True)

for i in tqdm(range(11188,30000)):
    img_file = 'sa_'+str(i)+'.jpg'
    json_file = img_file.replace('jpg', 'json')
    
    if not os.path.exists(dir+img_file):
        continue
    
    shutil.copy(dir+img_file, new_dir+img_file)
    shutil.copyfile(dir+img_file, new_dir+img_file)
    json_dict = json.loads(open(dir+json_file,'r').read())
    
    control_signal = np.zeros((json_dict['image']['height'],json_dict['image']['width'],3))
    annotations = json_dict['annotations']
    
    nums = len(annotations)
    length = 1<<24
    
    annotations = sorted(annotations, key=lambda x: x['bbox'][2]*x['bbox'][3], reverse=True)
    
    for j, annotation in enumerate(annotations):
        encode_mask = annotation['segmentation']
        decode_mask = mask.decode(encode_mask)
        
        pos = int((length-1)*(j+1)/nums)  
        color = (pos%(1<<8), pos//(1<<8)%(1<<8), pos//(1<<16))
        control_signal[decode_mask==1] = color

    alls = len(np.unique(control_signal[:,:,0] + control_signal[:,:,1]*256 + control_signal[:,:,2]*256*256))
    
    control_signal = control_signal[:,:,::-1]
    cv2.imwrite(new_dir+img_file.replace('.jpg','.png'),control_signal)
        
        
        
        
        
            
        
        
    
    
    
