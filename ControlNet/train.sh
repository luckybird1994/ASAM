#! /bin/bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python tutorial_train.py --batchsize=8 --gpus 8 --epoch 5 --dataset ../sam-1b/sa_000002  --json_path ../sam-1b/sa_000002-controlnet-train.json \
--resume=lightning_logs/version_0/checkpoints/epoch=4-step=874.ckpt