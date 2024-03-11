import os
import cv2
import shutil
from pycocotools  import mask
from pathlib import Path
import json
import numpy as np

dir = '/data/tanglv/data/sam-1b-subset/'
new_dir = '../sam-subset_11187/'

files = os.listdir(dir)
Path(new_dir).mkdir(exist_ok=True, parents=True)

for i in range(11187):
    img_file = 'sa_'+str(i)+'.jpg'
    json_file = img_file.replace('jpg', 'json')
    signal_file = 'sa_'+str(i)+'.png'
    
    if not os.path.exists(dir+img_file):
        continue
    

    json_dict = json.loads(open(dir+json_file,'r').read())
    
    control_signal = cv2.imread(new_dir+signal_file)
    
    annotations = json_dict['annotations']
    
    nums = len(annotations)
    
    control_signal = control_signal[...,2] + control_signal[...,1]*255 + control_signal[...,0]*255*255
    print(nums, control_signal.shape,np.unique(control_signal).shape)
    print(len((np.unique(control_signal)))   )
    print(nums ==  len(np.unique(control_signal)) )

        
        
        
        
        
            
        
        
    
    
    
