import json
import os
file = '/data/tanglv/data/sa_000001-blip2-caption.json'
list = []

new_file = '../ControlNet-main/sam-1b-controlnet-train_1.json'
new_f = open(new_file,'w')

with open(file,'r') as f:
    lines = f.readlines()
    for i,line in enumerate(lines):
        line = json.loads(line)
        line['target'] = line['img']
        line['source'] = line['img'][:-4]+'.png'
        line.pop('img')
        list.append(line)
        json.dump(line,new_f)
        if i < len(lines)-1: new_f.write('\n')
        

