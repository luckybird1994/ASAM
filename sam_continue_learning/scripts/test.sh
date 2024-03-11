export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30002  main.py \
--model-type vit_b \
--output_prefix sam_token-tuning_adv@4 \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--eval \
--prompt_type box \
--train-datasets dataset_sa000000adv_dice \
--valid-datasets dataset_hrsod_val dataset_ade20k_val dataset_voc2012_val dataset_cityscapes_val dataset_coco2017_val dataset_camo dataset_big_val dataset_BBC038v1 dataset_DOORS1 dataset_DOORS2 dataset_ZeroWaste dataset_ndis_train dataset_Plittersdorf_test dataset_egohos \
--restore-model work_dirs/sam_token-tuning_adv@4-dice-vit_b-11186/epoch_9.pth \
