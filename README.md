## ASAM: Boosting Segment Anything Model with Adversarial Tuning, CVPR2024

<font size=7><div align='center'>ASAM: Boosting Segment Anything Model with Adversarial Tuning</div></font>
<p align="center">
  <a href='https://github.com/luckybird1994/ASAM'><img src='https://img.shields.io/badge/Project-Page-Green'></a>
  <a href='http://arxiv.org/abs/2306.15195'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>
  <a href='https://huggingface.co/spaces/xhk/ASAM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Online Demo-green'></a>
  <a href='https://huggingface.co/xhk/ASAM'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoint-blue'></a>
  <a href='https://huggingface.co/datasets/xhk/ASAM-Datasets'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow'></a>
</p>

<div align=center><img width="70%" src="assets/framework.png"/></div>

## Abstract

In the evolving landscape of computer vision, foundation models have emerged as pivotal tools, exhibiting exceptional adaptability to a myriad of tasks. Among these, the Segment Anything Model (SAM) by Meta AI has distinguished itself in image segmentation. However, SAM, like its counterparts, encounters limitations in specific niche applications, prompting a quest for enhancement strategies that do not compromise its inherent capabilities. This paper introduces ASAM, a novel methodology that amplifies SAM's performance through adversarial tuning. We harness the potential of natural adversarial examples, inspired by their successful implementation in natural language processing (NLP). By utilizing a stable diffusion model, we augment a subset (1\%) of the SA-1B dataset, generating adversarial instances that are more representative of natural variations rather than conventional imperceptible perturbations. Our approach maintains the photorealism of adversarial examples and ensures alignment with original mask annotations, thereby preserving the integrity of the segmentation task. The fine-tuned ASAM demonstrates significant improvements across a diverse range of segmentation tasks without necessitating additional data or architectural modifications. The results of our extensive evaluations confirm that ASAM establishes new benchmarks in segmentation tasks, thereby contributing to the advancement of foundational models in computer vision.

## Performance
<div align=center><img width="100%" src="assets/performance.png"/></div>

## News
- [x] [2024.02.27] Paper is accepted by CVPR2024 and GitHub repo is created.
- [x] [2024.03.10] We release the training code, pre-trained models, datasets and online demo.

## ToDo
More kinds of ASAM (vit-large and vit-huge) will soon be open source.

## :fire: Highlight

**ASAM** is the enhanced version of SAM, improving the generalizability and keeping the original model structure. Without any pain, ASAM can replace the SAM anywhere for better performance. We release [ASAM checkpoint](https://huggingface.co/xhk/ASAM/tree/main) and [continuously updated checkpoints](https://huggingface.co/xhk/ASAM/tree/main) on huggingface. Just enjoy!

## Install

```bash
conda create -n ASAM python=3.9
conda activate ASAM
pip install -r requirements.txt
```


## Train
After preparing [data](docs/data.md) and [controlnet](docs/controlnet.md), you can project image to diffusion latent using the command:
```bash
export CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
export now=1
export interval=1300
for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python null_text_inversion.py \
    --save_root=output/sa_000000-Inversion \
    --data_root=sam-1b/sa_000000 \
    --control_mask_dir=sam-1b/sa_000000 \
    --caption_path=sam-1b/sa_000000-blip2-caption.json \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000001.pth \
    --guidence_scale=7.5 \
    --steps=10 \
    --ddim_steps=50 \
    --start=${now} \
    --end=$((now + interval))\ &
    now=$(expr $now + $interval) 
done
wait
```
or `bash scripts/inversion.sh`, where `--controlnet_path` is the checkpoint of pretrained controlnet and `--guidence_scale --steps --ddim_steps` are the diffusion-style arguments.

After preparing diffusion latent and [SAM checkpoint](https://github.com/facebookresearch/segment-anything?tab=readme-ov-file), you can optimize the latent towards the adversarial direction and get adversarial exampls using the command:
```bash
CUDA_VISIBLE_DEVICES_LIST=(0 1 2 3 4 5 6 7)
now=1
interval=1300
for id in "${CUDA_VISIBLE_DEVICES_LIST[@]}"
do
    echo "Start: ${now}"
    echo "End $((now + interval))"
    echo "GPU $id" 
    export CUDA_VISIBLE_DEVICES=${id} 
    python grad_null_text_inversion_edit.py \
    --save_root=output/sa_000001-Grad \
    --data_root=sam-1b/sa_000001 \
    --control_mask_dir=sam-1b/sa_000001 \
    --caption_path=sam-1b/sa_000001-blip2-caption.json \
    --inversion_dir=output/sa_000001-Inversion/embeddings \
    --controlnet_path=ckpt/control_v11p_sd15_mask_sa000002.pth \
    --eps=0.2 --steps=10 --alpha=0.02 \
    --mu=0.5 --beta=1.0 --norm=2 --gamma=100 --kappa=100 \
    --sam_batch=140 \
    --start=${now} \
    --end=$((now + interval)) 
    now=$(expr $now + $interval) 
done
```
or `bash scripts/grad.sh`, where `--mu=0.5  --beta --norm=2 --gamma --kappa` are the adversarial-style arguments.

After preparing the adversarial examples and [validation data](doc/data.md), you can tuning the
the sam using this command:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30011  main.py \
--model-type vit_b \
--output_prefix sam_token-tuning_adv0.1@4 \
--batch_size_train=4 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--learning_rate=1e-3 \
--train-datasets=dataset_sa000000adv_dice_0_1 \
--valid-datasets=dataset_hrsod_val \
--slow_start \
```
or `cd sam_continue_learning & bash scripts/train.sh`, where `--slow_start` is the learning rate trick. ASAM checkpoint will be save at `workdirs/sam_token-tuning_adv0.1@4` folder.

## Inference
After tuning the sam and preparing the [test data](docs/data.md), you can inference using this command:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.run --nproc_per_node=8 --master_port=30001  main.py \
--model-type vit_b \
--output_prefix work_dirs/diceloss_sam_iou_masktoken-tuning_b_adv@4 \
--batch_size_train=8 \
--batch_size_prompt=4 \
--batch_size_prompt_start=0 \
--find_unused_params \
--numworkers=0 \
--eval \
--prompt_type box \
--train-datasets dataset_sa000000 \
--valid-datasets dataset_hrsod_val dataset_camo \
--restore-model work_dirs/sam_token-tuning_adv0.1@4-1-vit_b-11186/epoch_9.pth \
```
or `cd sam_continue_learning & bash scripts/test.sh`, where `--restore-model` is the ASAM checkpoint path and `--valid-datasets` is the test datasets.


## Demo
We provide the [online demo](https://huggingface.co/spaces/xhk/ASAM) on huggingface, for example:
<div align=center><img width="90%" src="assets/demo.jpg"/></div>

## Cite
```bibtex
@article{tang2023can,
  title={Can sam segment anything? when sam meets camouflaged object detection},
  author={Tang, Lv and Xiao, Haoke and Li, Bo},
  journal={arXiv preprint arXiv:2304.04709},
  year={2023}
}
```

## Acknowledgement
This repo benefits from [SAM](https://github.com/facebookresearch/segment-anything) and [ControlNet](https://github.com/lllyasviel/ControlNet).  Our heartfelt gratitude goes to the developers of these resources!

## Contact   
Feel free to leave issues here or send our e-mails (libra@vivo.com, hk.xiao.me@gmail.com, luckybird1994@gmail.com).
### 