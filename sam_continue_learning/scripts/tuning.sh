export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  main.py \
    --model-type vit_b \
    --output_prefix asam_vit-b_tuning \
    --find_unused_params \
    --train-datasets=dataset_sa000000_adv \
    --valid-datasets=dataset_hrsod_val \
    --slow_start 