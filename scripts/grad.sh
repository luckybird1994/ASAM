#!/bin/bash
python grad_null_text_inversion_edit.py \
    --save_root=output/sa_000000-Grad \
    --data_root=SAM-1B/sa_000000 \
    --control_mask_dir=SAM-1B/sa_000000 \
    --caption_path=SAM-1B/sa_000000-blip2-caption.json \
    --inversion_dir=output/sa_000000-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_adv.pth \
    --eps=0.2 --steps=10 --alpha=0.02 \
    --mu=0.5 --beta=1.0 --norm=2 --gamma=100 --kappa=100 \
    --sam_batch=140 \
    --start=0 --end=11186 \
    --model_pth=ckpt/sam_vit_b_01ec64.pth
