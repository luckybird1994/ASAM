#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tutorial_train.py --batchsize=8 --gpus 8 --epoch 100 --dataset ../SAM-1B/sa_000000  --json_path ../SAM-1B/sa_000000-controlnet-train.json \
--resume models/control_sd15_ini.ckpt