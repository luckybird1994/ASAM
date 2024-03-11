export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=4 --master_port=30011  main.py \
--model-type vit_b \
--output_prefix sam_token-tuning_adv@4 \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=2e-3 \
--train-datasets=dataset_sa000000adv_dice \
--valid-datasets=dataset_hrsod_val \
--slow_start \