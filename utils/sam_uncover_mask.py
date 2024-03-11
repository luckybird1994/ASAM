import os
import cv2
import shutil
from pycocotools  import mask
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

dir = '/data/tanglv/data/sam-1b/sa_000001/'
new_dir = '/data/tanglv/data/sam-1b/sa_000001/'

files = os.listdir(dir)
Path(new_dir).mkdir(exist_ok=True, parents=True)

for i in tqdm(range(10000,30000)):
    img_file = 'sa_'+str(i)+'.jpg'
    json_file = img_file.replace('jpg', 'json')
    
    if not os.path.exists(dir+img_file): continue
    
    json_dict = json.loads(open(dir+json_file,'r').read())    
    annotations = json_dict['annotations']
    annotations = sorted(annotations, key=lambda x: x['bbox'][2]*x['bbox'][3], reverse=True)
    
    Path(os.path.join(new_dir, 'sa_'+str(i))).mkdir(parents=True, exist_ok=True)
    for j, annotation in enumerate(annotations):
        encode_mask = annotation['segmentation']
        decode_mask = mask.decode(encode_mask)*255.0
        decode_mask = decode_mask.astype(np.uint8)
        
        cv2.imwrite(os.path.join(new_dir, 'sa_'+str(i), 'segmentation_'+str(j)+'.png'), decode_mask)
        
        
        
        
        
            
        
        
    
    
    
