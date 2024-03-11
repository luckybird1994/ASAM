import torch
from PIL import Image
import os
from tqdm import tqdm, trange
import json
from lavis.models import load_model_and_preprocess

# setup device to use
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

#CUDA_VISIBLE_DEVICES=5 python generate_captin.py
dir = '/data/tanglv/data/'
datasets = ['sa_000001']

# loads BLIP-2 pre-trained model
model, vis_processors, _ = load_model_and_preprocess(name="blip2_t5", model_type="caption_coco_flant5xl", is_eval=True, device=device)

for dataset in datasets:
    dataset_dir = os.path.join(dir, dataset)
    
    prompt_path = dir + '/'+ dataset +'-blip2-caption.json'
    # if os.path.exists(prompt_path):
    #     os.remove(prompt_path)
    f = open(prompt_path,'a')
    
    files = os.listdir(dataset_dir)
    img_files = [x for x in files if 'jpg' in x]
    img_files = sorted(img_files)
        
    for i in trange(11187,30000):    
        img_file= 'sa_' + str(i+1)+'.jpg'
        print(img_file)
        
        if not os.path.exists(os.path.join(dataset_dir,img_file)):
            print(os.path.join(dataset_dir,img_file),'does not exist')
            continue
        
        raw_image = Image.open(os.path.join(dataset_dir,img_file)).convert("RGB")
        image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

        example = {}
        example["img"] =  img_file
        example["prompt"] = model.generate({"image": image})[0]
        
        json.dump(example,f)
        f.write(f"\n")