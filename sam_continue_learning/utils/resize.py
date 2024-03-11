import cv2
import os

dir = '/data/tanglv/xhk/Ad-Sam/2023-9-7/Ad-Sam-Main/HuggingFace-Demo/Space'

for file in os.listdir(dir):
    if '.jpg' not in file: continue
    img = cv2.imread(dir+'/'+file)
    img = cv2.resize(img, (1024, 1024))
    cv2.imwrite(dir+'/'+file, img)