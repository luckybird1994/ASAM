import os

dir = '/data/tanglv/Ad-SAM/2023-9-7/Ad-Sam-Main/output/sa_000000-Grad/skip-ablation-01-mi-0.5-sam-vit_b-150-0.01-100-1-2-10-Clip-0.2/record'
all_iou, cnt = 0, 0

for file in os.listdir(dir):
    if 'txt' not in file: continue
    
    all_iou += float(open(os.path.join(dir,file),'r').read())
    cnt += 1

print(all_iou/cnt)

with open(os.path.join(dir,'all.txt'),'w') as f:
    f.write(str(all_iou/cnt))