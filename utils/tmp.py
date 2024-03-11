import os
import cv2
import shutil
from pycocotools  import mask
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm



json_dict = json.loads(open('/data/tanglv/xhk/ASAM/2023-9-7/Ad-Sam-Main/000000000139.jpg.json','r').read())

# x = mask.decode(json_dict['objects'])
for object in json_dict['objects']:
    print(type(object))
    z = mask.decode(object)
    
print(json_dict.keys())
    
        
        
        
        
            
        
        
    
    
    
