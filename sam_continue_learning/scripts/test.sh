export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30002  main.py \
    --model-type vit_b \
    --eval \
    --prompt_type box \
    --train-datasets dataset_sa000000_adv \
    --valid-datasets dataset_coco2017_val \
    --restore-model work_dirs/asam_vit-b_tuning/asam_decoder_epoch_19.pth \
