import os
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import random
from typing import Dict, List, Tuple
from datetime import datetime
from segment_anything_training import sam_model_registry
from segment_anything_training.modeling import TwoWayTransformer, MaskDecoder
import torch.distributed as dist

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter
from utils.loss_mask import loss_masks
import utils.misc as misc
from torch.optim.lr_scheduler import LambdaLR, StepLR
from pathlib import Path

def lr_lambda(epoch):
    if epoch < args.warmup_epoch:
        return (epoch + 1) / args.warmup_epoch  # warm up 阶段线性增加
    else:
        return args.gamma ** (epoch-args.warmup_epoch+1) # warm up 后每个 epoch 除以 2

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x

class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x

class MaskDecoder_Tuning(MaskDecoder):
    def __init__(self, model_type):
        super().__init__(transformer_dim=256,
                        transformer=TwoWayTransformer(
                                depth=2,
                                embedding_dim=256,
                                mlp_dim=2048,
                                num_heads=8,
                            ),
                        num_multimask_outputs=3,
                        activation=nn.GELU,
                        iou_head_depth= 3,
                        iou_head_hidden_dim= 256,)
        """
        Predicts masks given an image and prompt embeddings, using a
        tranformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        assert model_type in ["vit_b","vit_l","vit_h"]
        
        checkpoint_dict = {"vit_b": os.path.join(args.load_prefix,"pretrained_checkpoint/sam_vit_b_maskdecoder.pth"),
                           "vit_l":"pretrained_checkpoint/sam_vit_l_maskdecoder.pth",
                           'vit_h':"pretrained_checkpoint/sam_vit_h_maskdecoder.pth"}
        checkpoint_path = checkpoint_dict[model_type]
        self.load_state_dict(torch.load(checkpoint_path))
        print("Tune Decoder init from SAM MaskDecoder")
        for n,p in self.named_parameters():
            if 'mask_tokens' not in n and 'iou_token' not in n:
                p.requires_grad = False
            else :
                print(p.shape)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """


        batch_len = len(image_embeddings)
        masks = []
        iou_preds = []
        for i_batch in range(batch_len):
            mask, iou_pred = self.predict_masks(
                image_embeddings=image_embeddings[i_batch].unsqueeze(0),
                image_pe=image_pe[i_batch],
                sparse_prompt_embeddings=sparse_prompt_embeddings[i_batch],
                dense_prompt_embeddings=dense_prompt_embeddings[i_batch],
            )
            masks.append(mask)
            iou_preds.append(iou_pred)
        masks = torch.cat(masks,0)
        iou_preds = torch.cat(iou_preds,0)
        
        
        # Select the correct mask or masks for output
        if multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        
        #print(self.iou_token.weight.shape, self.mask_tokens.weight.shape)
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
        #print(output_tokens.shape)
        #raise NameError
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0) 
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


def show_anns(labels_val, masks, input_point, input_box, input_label, filename, image, ious, boundary_ious):
    if len(masks) == 0:
        return

    #print(masks.shape, len(ious), len(boundary_ious))
    for i, (label_val,mask, iou, biou) in enumerate(zip(labels_val, masks, ious, boundary_ious)):
       
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(label_val/255.0, plt.gca(), label_mode=True) 
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'_gt.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close() 
        
        plt.figure(figsize=(10,10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        if input_box is not None:
            show_box(input_box[i], plt.gca())
        if (input_point is not None) and (input_label is not None): 
            show_points(input_point[i], input_label[i], plt.gca())
        plt.axis('off')
        plt.savefig(filename+'_'+str(i)+'.jpg',bbox_inches='tight',pad_inches=-0.1)
        plt.close()

def record_iou(filename, ious, boundary_ious):
    if len(ious) == 0:
        return

    for i, (iou, biou) in enumerate(zip(ious, boundary_ious)):
        with open(filename+'_'+str(i)+'.txt','w') as f:
            f.write(str(round(iou.item()*100,2)))
        
def show_points(coords, labels, ax, marker_size=175):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25) 

def show_mask(mask, ax, label_mode=False,random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif label_mode:
        color = np.array([122/255, 166/255, 82/255, 0.6])
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
      
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


def get_args_parser():
    parser = argparse.ArgumentParser('Tune-SAM', add_help=False)

    # Base Setting
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--compile', action='store_true')
    
    parser.add_argument('--numworkers', type=int, default=-1)
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument("--restore-sam-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")
    parser.add_argument('--train-datasets', nargs='+')
    parser.add_argument('--valid-datasets', nargs='+')
    parser.add_argument('--load_prefix', default='')
    
    # SAM setting
    parser.add_argument("--model-type", type=str, default="vit_l", 
                        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")
    parser.add_argument('--baseline', action='store_true')
    
    # Tuning Decoder setting
    parser.add_argument('--tuning_part', default='output_token', choices=['output_token','decoder'])

    # Base Learning Rate Setting & Epoch
    parser.add_argument('--learning_rate', default=5e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--max_epoch_num', default=10, type=int)
    
    # Step Learning Rate
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    
    # Slow start & Fast decay
    parser.add_argument('--warmup_epoch', default=5, type=int)
    parser.add_argument('--gamma', default=0.5, type=float)
    parser.add_argument('--slow_start', action='store_true')
    
    # Input Setting
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=1, type=int)
    parser.add_argument('--batch_size_prompt_start', default=0, type=int)
    parser.add_argument('--batch_size_prompt', default=-1, type=int)
    parser.add_argument('--train_img_num', default=11186, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--prompt_type', default='box')
    parser.add_argument('--point_num', type=int, default=10)
    
    # DDP Setting
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    # Output Setting
    parser.add_argument("--output_prefix", type=str, required=False, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--model_save_fre', default=1, type=int)
    return parser.parse_args()


def main(train_datasets, valid_datasets, args):
    ### --- Step 0: Initialize ---
    misc.init_distributed_mode(args)
    print('world size: {}'.format(args.world_size))
    print('rank: {}'.format(args.rank))
    print('local_rank: {}'.format(args.local_rank))
    print("args: " + str(args) + '\n')

    if misc.is_main_process():
        os.makedirs(args.output, exist_ok=True)
        with open(args.output+'/log.txt','a') as f:
            f.write('\n\n\n=========>> '+str(datetime.now())+'\n')
            f.write(str(args)+'\n')
            
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    ### --- Step 1: Train or Valid dataset ---
    if not args.eval:
        print("--- create training dataloader ---")
        train_im_gt_list = get_im_gt_name_dict(train_datasets, flag="train", limit=args.train_img_num)
        train_dataloaders, train_datasets = create_dataloaders(train_im_gt_list,
                                                        my_transforms = [
                                                                    RandomHFlip(),
                                                                    LargeScaleJitter()
                                                                    ],
                                                        batch_size = args.batch_size_train,
                                                        batch_size_prompt = args.batch_size_prompt,
                                                        batch_size_prompt_start = args.batch_size_prompt_start,
                                                        training = True,
                                                        numworkers=args.numworkers)
        print(len(train_dataloaders), " train dataloaders created")

    print("--- create valid dataloader ---")
    valid_im_gt_list = get_im_gt_name_dict(valid_datasets, flag="valid")
    valid_dataloaders, valid_datasets = create_dataloaders(valid_im_gt_list,
                                                          my_transforms = [
                                                                        Resize(args.input_size)
                                                                    ],
                                                          batch_size=args.batch_size_valid,
                                                          batch_size_prompt = args.batch_size_prompt,
                                                          batch_size_prompt_start = args.batch_size_prompt_start,
                                                          training=False)
    print(len(valid_dataloaders), " valid dataloaders created")
    
    ### --- Step 2: Model for DistributedDataParallel---
    net = MaskDecoder_Tuning(args.model_type) 
    if args.compile: net = torch.compile(net)
    if torch.cuda.is_available():
        net.cuda()
    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    net_without_ddp = net.module
    
    if args.restore_model:
        print("restore model from:", args.restore_model)
        if torch.cuda.is_available():
            net_without_ddp.load_state_dict(torch.load(args.restore_model))
        else:
            net_without_ddp.load_state_dict(torch.load(args.restore_model,map_location="cpu"))

    parameters_grad, parameters_no_grad = 0, 0
    for n,p in net_without_ddp.named_parameters():
        if p.requires_grad: parameters_grad += 1
        else: parameters_no_grad += 1
    print("parameters_grad Number:", parameters_grad)
    print("parameters_no_grad Number:", parameters_no_grad)
    
    
    sam_checkpoint_map = {
        'vit_b': os.path.join(args.load_prefix,'pretrained_checkpoint/sam_vit_b_01ec64.pth'),
        'vit_l': 'pretrained_checkpoint/sam_vit_l_0b3195.pth',
        'vit_h': 'pretrained_checkpoint/sam_vit_h_4b8939.pth',
    }
    sam = sam_model_registry[args.model_type](sam_checkpoint_map[args.model_type])
    if args.compile: sam = torch.compile(sam)
    _ = sam.to(device=args.device)
    sam = torch.nn.parallel.DistributedDataParallel(sam, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)
    if args.restore_sam_model:
        print("restore sam model from:", args.restore_sam_model)
        if torch.cuda.is_available():
            sam.module.load_state_dict(torch.load(args.restore_sam_model))
        else:
            sam.module.load_state_dict(torch.load(args.restore_sam_model,map_location="cpu"))
    
    
    ### --- Step 3: Train or Evaluate ---    
    if not args.eval:
        print("--- define optimizer ---")
        optimizer = optim.Adam(net_without_ddp.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)        
        if not args.slow_start:
            lr_scheduler = StepLR(optimizer, args.lr_drop_epoch)
            lr_scheduler.last_epoch=args.start_epoch
        else:
            print("slow start & fast decay")
            lr_scheduler = LambdaLR(optimizer, lr_lambda)

        train(args, net, sam, optimizer, train_dataloaders, valid_dataloaders, lr_scheduler)
    else:
        evaluate(args, net, sam, valid_dataloaders, args.visualize)


def train(args, net, sam,optimizer, train_dataloaders, valid_dataloaders, lr_scheduler):
    epoch_start = args.start_epoch
    epoch_num = args.max_epoch_num

    net.train()
    _ = net.to(device=args.device)
        
    for epoch in range(epoch_start,epoch_num): 
        print("epoch:   ",epoch, "  learning rate:  ", optimizer.param_groups[0]["lr"])
 
        metric_logger = misc.MetricLogger(delimiter="  ")
        train_dataloaders.batch_sampler.sampler.set_epoch(epoch)
    
        for data in metric_logger.log_every(train_dataloaders,10):
            
            inputs, labels = data['image'], data['label']  # [K 3 1024 1024]   [K N 1024 1024]
            K, N, H, W =labels.shape
            if torch.cuda.is_available(): 
                inputs = inputs.cuda()
                labels = labels.reshape(K*N,H,W).cuda()  #[K*N 1024 1024]
                
            imgs = inputs.permute(0, 2, 3, 1).cpu().numpy()  #[K 1024 1024 3]
            
            # input prompt
            input_keys = ['box','point','noise_mask']
            labels_box = misc.masks_to_boxes(labels) #[K*N 4]
            try:
                labels_points = misc.masks_sample_points(labels) #[K*N 10 2]
            except:
                # less than 10 points
                input_keys = ['box','noise_mask']
            labels_256 = F.interpolate(labels.unsqueeze(1), size=(256, 256), mode='bilinear')
            labels_noisemask = misc.masks_noise(labels_256) #[K*N 1 256 256]

            batched_input = []
            
            for bi in range(len(imgs)):
                dict_input = dict()
                input_image = torch.as_tensor(imgs[bi].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # [3 1024 1024]
                dict_input['image'] = input_image 

                input_type = random.choice(input_keys)
                sparse_slice, dense_slice = slice(bi*N,(bi+1)*N), slice(bi*N,(bi+1)*N)
                if input_type == 'box':
                    dict_input['boxes'] = labels_box[sparse_slice,...]  #N*4
                elif input_type == 'point':
                    point_coords = labels_points[sparse_slice,...] # N 10 2
                    dict_input['point_coords'] = point_coords
                    dict_input['point_labels'] = torch.ones(point_coords.shape[:-1], device=point_coords.device) #[N 10]
                elif input_type == 'noise_mask':
                    dict_input['mask_inputs'] = labels_noisemask[dense_slice] # N 1 256 256

                else:
                    raise NotImplementedError

                dict_input['original_size'] = imgs[0].shape[:2]
                
                batched_input.append(dict_input)
            with torch.no_grad():
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)
            
            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]

            masks = net(
                image_embeddings=encoder_embedding,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False
            )
            loss_mask, loss_dice = loss_masks(masks, labels.unsqueeze(1)/255.0, len(masks))
            loss = loss_mask + loss_dice
            
            loss_dict = {"loss_mask": loss_mask, "loss_dice":loss_dice}

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            losses_reduced_scaled = sum(loss_dict_reduced.values())
            loss_value = losses_reduced_scaled.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            metric_logger.update(training_loss=loss_value, **loss_dict_reduced)

        print("Finished epoch:      ", epoch)
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}

        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(f"Epoch {str(epoch)}: "+str(train_stats)[1:-1]+'\n')
        
        lr_scheduler.step()
        dist.barrier()
        test_stats = evaluate(args, net, sam, valid_dataloaders)
        train_stats.update(test_stats)
        
        net.train()  

        if epoch % args.model_save_fre == 0:
            model_name = "/epoch_"+str(epoch)+".pth"
            print('come here save at', args.output + model_name)
            misc.save_on_master(net.module.state_dict(), args.output + model_name)
    
    # Finish training
    print("Training Reaches The Maximum Epoch Number")
    
    # merge sam and tune_decoder
    if misc.is_main_process():        
        sam_checkpoint_map = {
        'vit_b': 'pretrained_checkpoint/sam_vit_b_01ec64.pth',
        'vit_l': 'pretrained_checkpoint/sam_vit_b_01ec64.pth',
        'vit_h': 'pretrained_checkpoint/sam_vit_b_01ec64.pth',
        }
        sam_ckpt = torch.load(sam_checkpoint_map[args.model_type]) 

        hq_decoder = torch.load(args.output + model_name)
        for key in hq_decoder.keys():
            if 'mask_token' in key or 'iou_token' in key:
                sam_key = 'mask_decoder.'+key
                sam_ckpt[sam_key] = hq_decoder[key]
        model_name = "/asam_epoch_"+str(epoch)+".pth"
        torch.save(sam_ckpt, args.output + model_name)

@torch.no_grad()
def compute_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    iou_list = []
    for i in range(0,len(preds)):
        single_iou = misc.mask_iou(postprocess_preds[i],target[i])
        iou = iou + single_iou
        iou_list.append(single_iou)
    return iou / len(preds), iou_list

@torch.no_grad()
def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    iou_list = []
    for i in range(0,len(preds)):
        single_iou = misc.boundary_iou(target[i],postprocess_preds[i])
        iou = iou + single_iou
        iou_list.append(single_iou)
    return iou / len(preds), iou_list

@torch.no_grad()
def evaluate(args, net, sam, valid_dataloaders, visualize=False):
    net.eval()
    print("Validating...")
    test_stats = {}
    dataset_id = -1
    
    for k in range(len(valid_dataloaders)):
        dataset_id += 1
        bad_examples = 0
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        print('valid_dataloader len:', len(valid_dataloader), valid_datasets[dataset_id]['name'])
        
        for data_val in metric_logger.log_every(valid_dataloader,10):
            
            imidx_val, inputs_val, labels_val, shapes_val, labels_ori, ori_im_path = data_val['imidx'], data_val['image'], data_val['label'], data_val['shape'], data_val['ori_label'],data_val['ori_im_path']
            K,N,H,W = labels_val.shape
            k,n,h,w = labels_ori.shape
            
            #print(inputs_val.shape, labels_val.shape, torch.max(labels_ori))
            
            # labels_val_np = labels_val[0,0,...].cpu().data.numpy()
            # cv2.imwrite("tmp.png",labels_val_np*255.0)
            # raise NameError
            if n == 0:
                bad_examples += 1
                loss_dict = {"val_iou_"+str(k): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(k): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue
            
            if torch.cuda.is_available():
                inputs_val = inputs_val.cuda()
                labels_val = labels_val.reshape(K*N,H,W).cuda() #K*N 1024 1024 
                labels_ori = labels_ori.reshape(k*n,h,w).cuda()
            
            imgs = inputs_val.permute(0, 2, 3, 1).cpu().numpy() # K 3 1024 1024 -> k 1024 1024 3
            
            if args.prompt_type=='box': 
                labels_box = misc.masks_to_boxes(labels_val) #K*N 4    
            if args.prompt_type=='point':        
                try:
                    labels_points = misc.masks_sample_points(labels_val,k=args.point_num) #[K*N 10 2]
                except:
                    bad_examples+=1
                    loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                    loss_dict_reduced = misc.reduce_dict(loss_dict)
                    metric_logger.update(**loss_dict_reduced)
                    continue
                    
            batched_input = []
            dict_input = dict()
            
            input_image = torch.as_tensor(imgs[0].astype(dtype=np.uint8), device=sam.device).permute(2, 0, 1).contiguous() # 3 1024 1024
            dict_input['image'] = input_image 
            if args.prompt_type == 'box':
                dict_input['boxes'] = labels_box #N 4
            elif args.prompt_type == 'point': 
                point_coords = labels_points #[N 10 2]
                #print(point_coords.shape)
                dict_input['point_coords'] = point_coords
                dict_input['point_labels'] = torch.ones(point_coords.size()[:2], device=point_coords.device)
            elif args.prompt_type == 'noise_mask':
                dict_input['mask_inputs'] = labels_noisemask[b_i:b_i+1]
            else:
                raise NotImplementedError
            
            dict_input['original_size'] = imgs[0].shape[:2]
            batched_input.append(dict_input)
            
            with torch.no_grad():            
                batched_output, interm_embeddings = sam(batched_input, multimask_output=False)

            batch_len = len(batched_output)
            encoder_embedding = torch.cat([batched_output[i_l]['encoder_embedding'] for i_l in range(batch_len)], dim=0)
            image_pe = [batched_output[i_l]['image_pe'] for i_l in range(batch_len)]
            sparse_embeddings = [batched_output[i_l]['sparse_embeddings'] for i_l in range(batch_len)]
            dense_embeddings = [batched_output[i_l]['dense_embeddings'] for i_l in range(batch_len)]
            
            if args.baseline:
                masks = batched_output[0]['low_res_logits'].to(torch.float32)
            else:
                masks = net(
                    image_embeddings=encoder_embedding,
                    image_pe=image_pe,
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
                masks = F.interpolate(masks, scale_factor=4, mode='bilinear', align_corners=False)
            #print(masks.shape,labels_ori.shape)
            
            try:
                iou,iou_list = compute_iou(masks,labels_ori.unsqueeze(1))
                boundary_iou,boundary_iou_list = compute_boundary_iou(masks,labels_ori.unsqueeze(1))
            except:
                bad_examples += 1
                loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue
            
            if torch.isnan(iou).any() or torch.isnan(boundary_iou).any():
                bad_examples += 1
                loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda(), "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): torch.tensor(0.5).cuda()}
                loss_dict_reduced = misc.reduce_dict(loss_dict)
                metric_logger.update(**loss_dict_reduced)
                continue    
        
            
            if args.prompt_type =='box':
                save_dir = os.path.join(args.output, args.prompt_type, valid_datasets[dataset_id]['name'])
            else:
                save_dir = os.path.join(args.output, args.prompt_type+'_'+str(args.point_num), valid_datasets[dataset_id]['name'])
            
            Path(save_dir).mkdir(parents=True,exist_ok=True)
            base = ori_im_path[0].split('/')[-1].split('.')[0]
            save_base = os.path.join(save_dir, str(base))
            record_iou(save_base, iou_list, boundary_iou_list)
            if visualize:
                masks_vis = (F.interpolate(masks.detach(), (1024, 1024), mode="bilinear", align_corners=False) > 0).cpu()
                imgs_ii = imgs[0].astype(dtype=np.uint8)
    
                if args.prompt_type=='box':
                    show_anns(labels_val.cpu(), masks_vis, None, labels_box.cpu(), None, save_base , imgs_ii, iou_list, boundary_iou_list)
                elif args.prompt_type=='point':
                    show_anns(labels_val.cpu(), masks_vis, labels_points.cpu(), None, torch.ones(labels_points.shape[:2]).cpu(), save_base , imgs_ii, iou_list, boundary_iou_list)
            
            loss_dict = {"val_iou_"+str(valid_datasets[dataset_id]['name']): iou, "val_boundary_iou_"+str(valid_datasets[dataset_id]['name']): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)
        
        
        print('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)
        print((str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples) +'\n') )
        
        text_log = {k: round(meter.global_avg*100,2) for k, meter in metric_logger.meters.items() if meter.count > 0}
        if misc.is_main_process():
            with open(args.output+'/log.txt','a') as f:
                f.write(str(valid_datasets[dataset_id]['name'])+' '+ str(text_log)[1:-1].replace("'","")+'\n')    
                f.write(str(valid_datasets[dataset_id]['name'])+' bad examples:'+ str(bad_examples) +'\n') 

    return test_stats


if __name__ == "__main__":

    ### --------------- Configuring the Train and Valid datasets ---------------    
    dataset_sa000000adv_dice = {"name": "sam_subset",
        "im_dir": "../output/sa_000000-Grad/skip-ablation-01-mi-SD-7.5-50-SAM-sam-vit_b-140-ADV-0.2-10-0.01-0.5-100.0-100.0-1.0-2/adv",
        "gt_dir": "../sam-1b/sa_000000",
        "im_ext": ".png",
        "gt_ext": ""}
    
    
    # valid set
    
    # single
    dataset_hrsod_val = {"name": "HRSOD-TE",
            "im_dir": "data/HRSOD-TE/imgs",
            "gt_dir": "data/HRSOD-TE/gts",
            "im_ext": ".jpg",
            "gt_ext": ".png"}

    
    dataset_ade20k_val = {"name": "ADE20K_2016_07_26",
            "im_dir": "data/ADE20K_2016_07_26/images/validation",
            "gt_dir": "data/ADE20K_2016_07_26/images/validation",
            "im_ext": ".jpg",
            "gt_ext": "_seg.png"}
    
    dataset_cityscapes_val = {"name": "cityscaps_val",
            "im_dir": "data/cityscapes/leftImg8bit/val",
            "gt_dir": "data/cityscapes/gtFine/val",
            "im_ext": "_leftImg8bit.png",
            "gt_ext": "_gtFine_instanceIds.png"}
    
    dataset_voc2012_val = {"name": "voc2012_val",
            "im_dir": "data/VOC2012/JPEGImages_val",
            "gt_dir": "data/VOC2012/SegmentationObject",
            "im_ext": ".jpg",
            "gt_ext": ".png"}
    #实列分割
    dataset_coco2017_val = {"name": "coco2017_val",
            "im_dir": "data/COCO2017-val/val2017",
            "annotation_file": "data/COCO2017-val/instances_val2017.json",
            "im_ext": ".jpg"
            }
    dataset_camo = {"name": "camo",
        "im_dir": "data/CAMO/imgs",
        "gt_dir": "data/CAMO/gts",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ishape_antenna = {"name": "ishape",
        "im_dir": "data/ishape_dataset/antenna/val/image",
        "gt_dir": "data/ishape_dataset/antenna/val/instance_map",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "data/Plant_Phenotyping_Datasets",
        "gt_dir": "data/Plant_Phenotyping_Datasets",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
        }
    
    dataset_gtea_train = {"name": "gtea",
            "im_dir": "data/GTEA_hand2k/GTEA_GAZE_PLUS/Images",
            "gt_dir": "data/GTEA_hand2k/GTEA_GAZE_PLUS/Masks",
            "im_ext": ".jpg",
            "gt_ext": ".png"
    }
    
    dataset_streets = {"name": "streets_coco",
        "im_dir": "data/vehicleannotations/images",
        "annotation_file": "data/vehicleannotations/annotations/vehicle-annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_TimberSeg = {"name": "timberseg_coco",
        "im_dir": "data/y5npsm3gkj-2/prescaled/",
        "annotation_file": "data/y5npsm3gkj-2/prescaled/coco_annotation_rotated.json",
        "im_ext": ".png",
    }
    
    dataset_ppdls = {"name": "ppdls",
        "im_dir": "data/Plant_Phenotyping_Datasets",
        "gt_dir": "data/Plant_Phenotyping_Datasets",
        "im_ext": "_rgb.png",
        "gt_ext": "_label.png"
        }
    
    dataset_gtea_train = {"name": "gtea",
        "im_dir": "data/GTEA_GAZE_PLUS/Images",
        "gt_dir": "data/GTEA_GAZE_PLUS/Masks",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_streets = {"name": "streets_coco",
        "im_dir": "data/vehicleannotations/images",
        "annotation_file": "data/vehicleannotations/annotations/vehicle-annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_big_val = {"name": "big",
        "im_dir": "data/BIG/val",
        "gt_dir": "data/BIG/val",
        "im_ext": "_im.jpg",
        "gt_ext": "_gt.png"
    }
    
    dataset_ndis_train = {"name": "ndis_park_coco",
        "im_dir": "data/ndis_park/train/imgs",
        "annotation_file": "data/ndis_park/train/train_coco_annotations.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_test = {"name": "Plittersdorf_coco",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/test.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_train = {"name": "Plittersdorf_coco",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/train.json",
        "im_ext": ".jpg",
    }
    
    dataset_Plittersdorf_val = {"name": "Plittersdorf_coco",
        "im_dir": "data/plittersdorf_instance_segmentation_coco/images",
        "annotation_file": "data/plittersdorf_instance_segmentation_coco/val.json",
        "im_ext": ".jpg",
    }
    
    dataset_egohos = {"name": "egohos",
        "im_dir": "data/egohos/val/image",
        "gt_dir": "data/egohos/val/label",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_LVIS = {"name": "LVIS",
        "im_dir": "data/LVIS/val2017",
        "annotation_file": "data/LVIS/annotations/lvis_v1_val.json",
        "im_ext": ".jpg",
    }
    dataset_BBC038v1 = {"name": "BBC038v1",
        "im_dir": "data/BBC038V1-Train",
        "annotation_file": "data/BBC038V1-Train",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS1 = {"name": "DOORS1",
        "im_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/img",
        "gt_dir": "data/DOORS/Regression/Te1_5000_b_2022-08-02 11.16.00/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    dataset_DOORS2 = {"name": "DOORS2",
        "im_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/img",
        "gt_dir": "data/DOORS/Regression/Te2_5000_ub_2022-08-02 11.16.11/Rock_all",
        "im_ext": ".png",
        "gt_ext": ".png"
    }
    
    
    dataset_NDD20_ABOVE = {"name": "NDD20_coco",
        "im_dir": "data/NDD20/ABOVE",
        "annotation_file": "data/NDD20/ABOVE_LABELS.json",
        "im_ext": ".jpg",
    }
        
    dataset_PIDRAY = {"name": "pid_coco",
        "im_dir": "data/pidray/hard",
        "annotation_file": "data/pidray/annotations/xray_test_hard.json",
        "im_ext": ".jpg",
    }
    
    dataset_TrashCan_val = {"name": "TrashCAN_coco",
        "im_dir": "data/TrashCan/instance_version/val",
        "annotation_file": "data/TrashCan/instance_version/instances_val_trashcan.json",
        "im_ext": ".jpg",
    }
    
    dataset_ZeroWaste = {"name": "ZeroWaste",
        "im_dir": "data/splits_final_deblurred/train/data",
        "gt_dir": "data/splits_final_deblurred/train/sem_seg",
        "im_ext": ".PNG",
        "gt_ext": ".PNG"
    }
    
    dataset_DRAM_test = {"name": "DRAM",
        "im_dir": "data/DRAM_raw",
        "gt_dir": "data/DRAM_raw",
        "im_ext": ".jpg",
        "gt_ext": ".png"
    }
    
    dataset_ovis_train = {"name": "ovis_train_coco",
            "im_dir": "data/OVIS/train",
            "annotation_file": "data/OVIS/annotations_train.json",
            "im_ext": ".jpg"
    }
    
    args = get_args_parser()
    if not args.eval:
        args.output = os.path.join('work_dirs', args.output_prefix+'-'+args.train_datasets[0].split('_')[-1]+'-'+args.model_type+'-'+str(args.train_img_num))
    elif args.baseline: 
        if not args.restore_sam_model: args.output = os.path.join('work_dirs', args.output_prefix+'-'+args.model_type)
        else: args.output = os.path.join(*args.restore_sam_model.split('/')[:-1])
    elif args.restore_model:
        args.output = os.path.join(*args.restore_model.split('/')[:-1])
    
    # print(args.output)
    # raise NameError
    train_datasets = [globals()[dataset] for dataset in args.train_datasets]
    valid_datasets = [globals()[dataset] for dataset in args.valid_datasets]
    
    for train_dataset in train_datasets:
        train_dataset['im_dir'] = os.path.join(args.load_prefix, train_dataset['im_dir'])
        if 'gt_dir' in train_dataset: train_dataset['gt_dir'] = os.path.join(args.load_prefix, train_dataset['gt_dir'])
        if 'annotation_file' in train_dataset: train_dataset['annotation_file'] = os.path.join(args.load_prefix, train_dataset['annotation_file'])
        
    for test_dataset in valid_datasets:
        test_dataset['im_dir'] = os.path.join(args.load_prefix, test_dataset['im_dir'])
        if 'gt_dir' in test_dataset: test_dataset['gt_dir'] = os.path.join(args.load_prefix, test_dataset['gt_dir'])
        if 'annotation_file' in test_dataset: test_dataset['annotation_file'] = os.path.join(args.load_prefix, test_dataset['annotation_file'])
    
    main(train_datasets, valid_datasets, args)
