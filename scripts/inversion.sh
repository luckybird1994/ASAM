#!/bin/bash
python null_text_inversion.py \
    --save_root=output/sa_000000-Inversion \
    --data_root=SAM-1B/sa_000000 \
    --control_mask_dir=SAM-1B/sa_000000 \
    --caption_path=SAM-1B/sa_000000-blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_inv.pth \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=0 \
    --end=11187  