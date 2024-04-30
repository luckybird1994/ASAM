import pdb
import os
import argparse
from get_model import get_model
from typing import Optional, Union, Tuple, List, Dict
from tqdm import tqdm, trange
import torch
from diffusers import StableDiffusionControlNetPipeline, DDIMScheduler
from diffusers import ControlNetModel
import torch.nn.functional as nnf
import numpy as np
import abc
import ptp_utils
import seq_aligner
from PIL import Image
import time
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import cv2
import json    
from tqdm import tqdm
import matplotlib.pyplot as plt
from show import show_box, show_mask 

'''
CUDA_VISIBLE_DEVICES=0 python3 grad_null_text_inversion_edit.py --model sam --beta 1 --alpha 0.01 --steps 10  --ddim_steps=50 --norm 2
'''

############## Initialize #####################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# sam setting
parser.add_argument('--model', type=str, default='sam', help='cnn')
parser.add_argument('--model_type', type=str, default='vit_b', help='cnn')
parser.add_argument('--model_pth', type=str, default='vit_b', help='cnn')
parser.add_argument('--sam_batch', type=int, default=150, help='cnn')

# SD setting
parser.add_argument('--ddim_steps', default=50, type=int, help='random seed')
parser.add_argument('--guess_mode', action='store_true')   
parser.add_argument('--guidance_scale', default=7.5, type=float, help='random seed') 
parser.add_argument('--random_latent', action='store_true')

# grad setting
parser.add_argument('--alpha', type=float, default=0.01, help='cnn')
parser.add_argument('--gamma', type=float, default=100, help='cnn')
parser.add_argument('--kappa', type=float, default=100, help='cnn')
parser.add_argument('--beta', type=float, default=1, help='cnn')
parser.add_argument('--eps', type=float, default=0.2, help='cnn')
parser.add_argument('--steps', type=int, default=10, help='cnn')
parser.add_argument('--norm', type=int, default=2, help='cnn')
parser.add_argument('--mu', default=0.5, type=float, help='random seed')

# base setting
parser.add_argument('--start', default=1, type=int, help='random seed')
parser.add_argument('--end', default=11187, type=int, help='random seed')
parser.add_argument('--seed', default=0, type=int, help='random seed')
parser.add_argument('--check_controlnet', action='store_true')
parser.add_argument('--check_inversion', action='store_true')
parser.add_argument('--debug', action='store_true')

# path setting
parser.add_argument('--prefix', type=str, default='skip-ablation-01-mi', help='cnn')
parser.add_argument('--data_root', default='sam-1b/sa_000000', type=str, help='random seed')   
parser.add_argument('--save_root', default='output/sa_000000-Grad', type=str, help='random seed')   
parser.add_argument('--control_mask_dir', default='sam-1b/sa_000000', type=str, help='random seed')   
parser.add_argument('--inversion_dir', default='output/sa_000000-Inversion/embeddings', type=str, help='random seed')   
parser.add_argument('--caption_path', default='sam-1b/sa_000000-blip2-caption.json', type=str, help='random seed')    
parser.add_argument('--controlnet_path', default='ckpt/control_v11p_sd15_mask_sa000000.pth', type=str, help='random seed')    

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
torch.Generator().manual_seed(args.seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('==> Preparing Model..')
image_size = (1024, 1024)

if args.model == 'vit' or args.model == 'adv_resnet152_denoise':
    print('Using 0.5 Nor...')
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
elif args.model == 'mvit':
    mean = [0, 0, 0]
    std = [1, 1, 1] 
    image_size = (320, 320)
else:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

mean = torch.Tensor(mean).cuda()
std = torch.Tensor(std).cuda()

net = get_model(args.model, args.model_type, args.model_pth)


if device == 'cuda':
    net.to(device)
    cudnn.benchmark = True
net.eval()
net.cuda()

def str2img(value):
    width, height = 512, 512
    background_color = (255, 255, 255)  # 白色背景
    image = cv2.UMat(np.ones((height, width, 3), dtype=np.uint8) * background_color)
    
    # 在图像上绘制文本
    text = "worst_iou: " + str(value)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    text_color = (0, 0, 0)  # 黑色文本颜色
    thickness = 1

    # 获取文本的尺寸
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    x = (width - text_size[0]) // 2  # 计算文本的起始x坐标
    y = (height + text_size[1]) // 2  # 计算文本的起始y坐标

    # 在图像上绘制文本
    cv2.putText(image, text, (x, y), font, font_scale, text_color, thickness)

    return image.get()

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)
    
    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)
    y = y.to(masks)
    x = x.to(masks)

    x_mask = ((masks>128) * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    y_mask = ((masks>128) * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks>128), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """

    inputs = inputs.flatten(1)
    targets = targets.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.mean()

class LocalBlend:
    def get_mask(self, maps, alpha, use_pool):
        k = 1
        maps = (maps * alpha).sum(-1).mean(1)
        if use_pool:
            maps = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.th[1-int(use_pool)])
        mask = mask[:1] + mask
        return mask
    
    def __call__(self, x_t, attention_store):
        self.counter += 1
        if self.counter > self.start_blend:
           
            maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
            maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
            maps = torch.cat(maps, dim=1)
            mask = self.get_mask(maps, self.alpha_layers, True)
            if self.substruct_layers is not None:
                maps_sub = ~self.get_mask(maps, self.substruct_layers, False)
                mask = mask * maps_sub
            mask = mask.float()
            x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], substruct_words=None, start_blend=0.2, th=(.3, .3)):
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        
        if substruct_words is not None:
            substruct_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, (prompt, words_) in enumerate(zip(prompts, substruct_words)):
                if type(words_) is str:
                    words_ = [words_]
                for word in words_:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    substruct_layers[i, :, :, :, :, ind] = 1
            self.substruct_layers = substruct_layers.to(device)
        else:
            self.substruct_layers = None
        self.alpha_layers = alpha_layers.to(device)
        self.start_blend = int(start_blend * NUM_DDIM_STEPS)
        self.counter = 0 
        self.th=th


class EmptyControl:
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


@torch.no_grad()
def compute_iou(preds, target): #[N 1 h w] [-1 1] [0 1]
    def mask_iou(pred_label,label):
        '''
        calculate mask iou for pred_label and gt_label
        '''

        pred_label = (pred_label>0)[0].int()
        label = (label>128)[0].int()

        intersection = ((label * pred_label) > 0).sum()
        union = ((label + pred_label) > 0).sum()
        return intersection / union
    
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)
    
class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

class SpatialReplace(EmptyControl):
    
    def step_callback(self, x_t):
        if self.cur_step < self.stop_inject:
            b = x_t.shape[0]
            x_t = x_t[:1].expand(b, *x_t.shape[1:])
        return x_t

    def __init__(self, stop_inject: float):
        super(SpatialReplace, self).__init__()
        self.stop_inject = int((1 - stop_inject) * NUM_DDIM_STEPS)
        

class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

        
class AttentionControlEdit(AttentionStore, abc.ABC):
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace, place_in_unet):
        if att_replace.shape[2] <= 32 ** 2:
            attn_base = attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
            return attn_base
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce, place_in_unet)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        # attn_replace = attn_replace / attn_replace.sum(-1, keepdims=True)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(1, 77)
    
    for word, val in zip(word_select, values):
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = val
    return equalizer

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    out = out.sum(0) / out.shape[0]
    return out.cpu()


def make_controller(prompts: List[str], is_replace_controller: bool, cross_replace_steps: Dict[str, float], self_replace_steps: float, blend_words=None, equilizer_params=None) -> AttentionControlEdit:
    if blend_words is None:
        lb = None
    else:
        lb = LocalBlend(prompts, blend_word)
    if is_replace_controller:
        controller = AttentionReplace(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    else:
        controller = AttentionRefine(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps, self_replace_steps=self_replace_steps, local_blend=lb)
    if equilizer_params is not None:
        eq = get_equalizer(prompts[1], equilizer_params["words"], equilizer_params["values"])
        controller = AttentionReweight(prompts, NUM_DDIM_STEPS, cross_replace_steps=cross_replace_steps,
                                       self_replace_steps=self_replace_steps, equalizer=eq, local_blend=lb, controller=controller)
    return controller


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    ptp_utils.view_images(np.stack(images, axis=0), prefix='cross_attention')
    

def show_self_attention_comp(attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    attention_maps = aggregate_attention(attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), prefix='self_attention')


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    mask_control: Optional[torch.tensor] = None,
    uncond_embeddings=None,
    start_time=50,
    return_type='image'
):
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)
    for i, t in enumerate(tqdm(model.scheduler.timesteps[-start_time:])):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, mask_control, context, t, guidance_scale, low_resource=False, guess_mode=args.guess_mode)
        
    if return_type == 'image':
        image = ptp_utils.latent2image(model.vae, latents)
    else:
        image = latents
    return image, latent

def run_and_display(prompts, controller, latent=None, mask_control=None, run_baseline=False, generator=None, uncond_embeddings=None, verbose=True, prefix='inversion'):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, mask_control=mask_control, run_baseline=False, generator=generator)
        print("with prompt-to-prompt")
    images, x_t = text2image_ldm_stable(ldm_stable, prompts, controller, latent=latent, mask_control=mask_control,num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator, uncond_embeddings=uncond_embeddings)
    if verbose:
        ptp_utils.view_images(images, prefix=prefix)
    return images, x_t

def limitation01(y):
    idx = (y > 1)
    y[idx] = (torch.tanh(1000*(y[idx]-1))+10000)/10001
    idx = (y < 0)
    y[idx] = (torch.tanh(1000*(y[idx])))/10000
    return y

@torch.no_grad()
def text2image_ldm_stable_last(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    start_time=50,
    boxes=None,
    label_masks_256=None,
    mask_control=None,
    raw_img=None,
):
    
    batch_size = len(prompt)
    ptp_utils.register_attention_control(model, controller)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = ptp_utils.init_latent(latent, model, height, width, generator, batch_size)
    #print("Latent", latent.shape, "Latents", latents.shape) # [1 4 64 64]

    model.scheduler.set_timesteps(num_inference_steps)

    best_latent = latents
    ori_latents = latents.clone().detach()
    adv_latents = latents.clone().detach()
    #print(latents.max(), latents.min())
    momentum = 0
    worst_iou = 1.0
    worst_mask = None
    for k in range(args.steps):
        latents = adv_latents
        for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
            if uncond_embeddings_ is None:
                context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
            else:
                context = torch.cat([uncond_embeddings_, text_embeddings])
            #print(model.device, latents.device, mask_control.device, context.device, t.device)
            latents = ptp_utils.diffusion_step(model, controller, latents, mask_control, context, t, guidance_scale, low_resource=False, guess_mode=args.guess_mode)

        image = None
        with torch.enable_grad():
            latents_last = latents.detach().clone()
            latents_last.requires_grad = True
            latents_t = (1 / 0.18215 * latents_last)
            image = model.vae.decode(latents_t)['sample']
            image = (image / 2 + 0.5)
            image = limitation01(image)
            image_m = F.interpolate(image, image_size)

            if args.model == 'sam':
                example = {}
                example['image'] = image_m[0]*255.0
                example['boxes'] = boxes
                example['original_size'] = image_size
                output, interbeddings = net([example], multimask_output=False)
                output = output[0]
                ad_masks, ad_iou_predictions, ad_low_res_logits = output['masks'],output['iou_predictions'],output['low_res_logits']  
            elif args.model == 'sam_efficient':

                input_points = boxes.reshape(1,-1,2,2)
                input_labels = torch.concat([torch.full((1,boxes.shape[0],1),3), torch.full((1,boxes.shape[0],1),4)] ,-1).cuda()        
                predicted_logits, predicted_iou = net(
                    image_m,
                    input_points,
                    input_labels,
                )
                sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
                predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
                predicted_logits = torch.take_along_dim(
                    predicted_logits, sorted_ids[..., None, None], dim=2
                )
                ad_low_res_logits = F.interpolate(predicted_logits[0,:,:1,:,:],(256,256),mode='bilinear',align_corners=False)
                ad_masks = torch.ge(ad_low_res_logits,0)
                
            loss_ce = args.gamma * torch.nn.functional.binary_cross_entropy_with_logits(ad_low_res_logits, label_masks_256/255.0) 
            loss_dice = args.kappa * dice_loss(ad_low_res_logits.sigmoid(), label_masks_256/255.0)     

            iou = compute_iou(ad_low_res_logits, label_masks_256).item()
            print("iou:",iou)
            if iou < worst_iou:                
                best_latent, worst_iou, worst_mask  = adv_latents, iou, F.interpolate(ad_masks.to(torch.float32), size=(512,512), mode='bilinear', align_corners=False)
                    
            image_m = image_m - mean[None,:,None,None]
            image_m = image_m / std[None,:,None,None]
            loss_mse =  args.beta * torch.norm(image_m-raw_img, p=args.norm).mean()  # **2 / 50
            
            loss = loss_dice + loss_ce - loss_mse
            loss.backward()
            print('*' * 50)
            print('Loss', loss.item(), 'Loss_dice', loss_dice.item(),'Loss_ce', loss_ce.item(), 'Loss_mse', loss_mse.item())
            print(k, 'Predicted:', loss)
            print('Grad:', latents_last.grad.min(), latents_last.grad.max())
        
        l1_grad = latents_last.grad / torch.norm(latents_last.grad, p=1)
        print('L1 Grad:', l1_grad.min(), l1_grad.max())
        momentum = args.mu * momentum + l1_grad
        adv_latents = adv_latents + torch.sign(momentum) * args.alpha
        noise = (adv_latents - ori_latents).clamp(-args.eps, args.eps)
        adv_latents = ori_latents + noise
        latents = adv_latents.detach()

    # Return Best Attack
    latents = best_latent
    for i, t in enumerate(model.scheduler.timesteps[-start_time:]):
        if uncond_embeddings_ is None:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings_, text_embeddings])
        latents = ptp_utils.diffusion_step(model, controller, latents, mask_control,context, t, guidance_scale, low_resource=False)
        
    latents = (1 / 0.18215 * latents)
    image = model.vae.decode(latents)['sample']
    image = (image / 2 + 0.5)
    image = limitation01(image)
    print(2, image.max(), image.min())

    image = image.clamp(0, 1).detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    worst_mask_show = np.zeros((512,512,3))
    fig, ax = plt.subplots(figsize=(5.12, 5.12))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=0)
    ax.axis('off')
    plt.imshow(image[0]/255)
    
    if args.steps:
        num = origin_len
        length = 1<<24
        worst_mask = torch.concat([worst_mask,sup_masks/255])
        for i, single_mask in enumerate(worst_mask):
            single_mask = single_mask[0].cpu().numpy()
            pos = int((length-1) *(i+1) / num)
            color = np.array((pos%256, pos//256%256, pos//(1<<16)))
            worst_mask_show[single_mask!=0] = color
            
            if i < args.sam_batch:
                box= tuple((boxes[i].cpu().numpy()/2).tolist())        
                show_box(box, ax=ax, color=(color[0]/256, color[1]/256,color[2]/256))
            show_mask(single_mask, ax=ax, color=np.array((color[0]/256, color[1]/256,color[2]/256,0.4)))
        
    
    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    vis = np.frombuffer(data, dtype=np.uint8).reshape((height, width, 3))
    
    return image, best_latent, worst_mask_show, vis, worst_iou

def check_controlnet():
    id = 2    
    prompt = captions[f'sa_{str(id)}.jpg']

    control_image = Image.open(os.path.join(args.control_mask_dir, f'sa_{str(id)}.png'))
    
    output = ldm_stable(
        prompt, image=control_image,num_inference_steps=args.ddim_steps, guidance_scale=args.guidance_scale, guess_mode=args.guess_mode
    ).images[0]
    
    output_numpy = np.array(output)
    output.save('check_controlnet_pth.png')
    
    control_image = Image.open(os.path.join(args.control_mask_dir, f'sa_{str(id)}.png'))
    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512,512)) / 255.0
    control_image = torch.from_numpy(control_image).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
    controller = EmptyControl()
    x_t = torch.randn((1, ldm_stable.unet.in_channels,  512// 8, 512 // 8))
    run_and_display(prompts=[prompt], controller=controller, run_baseline=False, latent=x_t, mask_control=control_image,uncond_embeddings=None, verbose=True, prefix='check_controlnet_use')

def check_inversion():
    id = 1    
    prompt = captions[f'sa_{str(id)}.jpg']

    latent_path = f"{args.inversion_dir}/sa_{str(id)}_latent.pth"
    uncond_path = f"{args.inversion_dir}/sa_{str(id)}_uncond.pth"
    x_t = torch.load(latent_path).cuda()
    uncond_embeddings = torch.load(uncond_path).cuda()
    
    control_image = Image.open(os.path.join(args.control_mask_dir, f'sa_{str(id)}.png'))
    control_image = np.array(control_image)
    control_image = cv2.resize(control_image, (512,512)) / 255.0
    control_image = torch.from_numpy(control_image).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda()
    controller = AttentionStore()
    
    run_and_display(prompts=[prompt], controller=controller, run_baseline=False, latent=x_t, mask_control=control_image,uncond_embeddings=uncond_embeddings, verbose=True, prefix='check_inversion')


if __name__ == '__main__':
    # Load Stable Diffusion & ControlNet
    scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    MY_TOKEN = "your huggingface token"
    LOW_RESOURCE = False 
    NUM_DDIM_STEPS = args.ddim_steps
    GUIDANCE_SCALE = args.guidance_scale
    MAX_NUM_WORDS = 77
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    controlnet = ControlNetModel.from_single_file(args.controlnet_path).to(device)    
    ldm_stable = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", use_auth_token=MY_TOKEN,controlnet=controlnet, scheduler=scheduler).to(device)
    
    try:
        ldm_stable.disable_xformers_memory_efficient_attention()
    except AttributeError:
        print("Attribute disable_xformers_memory_efficient_attention() is missing")    
    
    # Load imgs and Caption  
    captions = {}
    with open(args.caption_path,'r') as f:
        lines = f.readlines()
        for line in lines:
            json_dict = json.loads(line.strip()) 
            captions[json_dict['img'].strip()] = json_dict['prompt'].strip()   
    
    # Check controlnet & inversion
    if args.check_controlnet: check_controlnet()
    if args.check_inversion: check_inversion()
    if args.check_inversion or args.check_controlnet: raise NameError 
    
    # Prepare save path
    save_path = args.save_root + '/' + args.prefix + '-SD-' + str(args.guidance_scale) + '-' +str(args.ddim_steps) +'-SAM-' + args.model + '-' + args.model_type +'-'+ str(args.sam_batch)+ '-ADV-' + str(args.eps) + '-' +str(args.steps)  + '-' + str(args.alpha) + '-' + str(args.mu)+  '-' +  str(args.kappa) +'-'+ str(args.gamma) + '-' + str(args.beta) + '-' + str(args.norm) 
    if args.random_latent:
        save_path += '-random_latent'
    
    print("Save Path:", save_path)
    if not os.path.exists(args.save_root): os.mkdir(args.save_root)
    if not os.path.exists(save_path): os.mkdir(save_path)
    if not os.path.exists(os.path.join(save_path,'pair')): os.mkdir(os.path.join(save_path,'pair'))
    if not os.path.exists(os.path.join(save_path,'adv')): os.mkdir(os.path.join(save_path,'adv'))
    if not os.path.exists(os.path.join(save_path,'record')): os.mkdir(os.path.join(save_path,'record'))
    
    # Adversarial grad loop
    for i in trange(args.start, args.end+1):
        
        # prepare img & mask path
        img_path = args.data_root+'/'+'sa_'+str(i)+'.jpg'
        control_mask_path = args.control_mask_dir+'/'+'sa_'+str(i)+'.png'
        label_mask_dir = args.data_root+'/'+'sa_'+str(i)
        if not os.path.exists(img_path):
            print(img_path, "does not exist!")
            continue
        
        if os.path.exists(os.path.join(save_path, 'adv', 'sa_'+str(i)+'.png')) and not args.debug:
            print(os.path.join(save_path, 'adv', 'sa_'+str(i)+'.png'), " has existed!")
            continue
        
        # load raw img for mse [1,3,512,512] [0,1]
        pil_image = Image.open(img_path).convert('RGB').resize(image_size)
        raw_img_show = np.array(pil_image.resize((512,512)))
        raw_img = (torch.tensor(np.array(pil_image).astype(np.float32), device=device).unsqueeze(0)/255.).permute(0,3, 1, 2)
        raw_img = raw_img - mean[None,:,None,None]
        raw_img = raw_img / std[None,:,None,None]
        
        # load mask labels & boxes prompt 
        global origin_len
        origin_len = len(os.listdir(label_mask_dir))
        label_masks = torch.empty([0,1,1024,1024]).cuda()
        for j in range(min(args.sam_batch,origin_len)):
            label_mask_path = os.path.join(label_mask_dir,f'segmentation_{str(j)}.png')
            label_mask = Image.open(label_mask_path).convert('L').resize((image_size))
            label_mask_torch = torch.tensor(np.array(label_mask)).cuda().to(torch.float32)
            label_masks = torch.cat([label_masks,label_mask_torch.unsqueeze(0).unsqueeze(0)])
        
        # load sup mask for show
        sup_masks = torch.empty([0,1,512,512]).cuda()
        for j in range(min(args.sam_batch,origin_len),origin_len):
            label_mask_path = os.path.join(label_mask_dir,f'segmentation_{str(j)}.png')
            label_mask = Image.open(label_mask_path).convert('L').resize((512,512)) 
            label_mask_torch = torch.tensor(np.array(label_mask)).cuda().to(torch.float32)
            sup_masks = torch.cat([sup_masks,label_mask_torch.unsqueeze(0).unsqueeze(0)])
        
        boxes = masks_to_boxes(label_masks.squeeze())    
        
        label_masks_256 = F.interpolate(label_masks, size=(256,256), mode='bilinear', align_corners=False) 
        
        # load caption
        prompt = captions[img_path.split('/')[-1]]
        print(prompt)
        
        # load x_t & uncondition embeddings 
        latent_path = f"{args.inversion_dir}/sa_{i}_latent.pth"
        uncond_path = f"{args.inversion_dir}/sa_{i}_uncond.pth"
        
        if not os.path.exists(latent_path) or not os.path.exists(uncond_path):
            print(latent_path, uncond_path, "do not exist!")
            continue
        else:
            x_t = torch.load(latent_path).cuda()
            uncond_embeddings = torch.load(uncond_path).cuda()
        
        if args.random_latent:
            x_t = torch.randn_like(x_t)
            uncond_embeddings = torch.randn_like(uncond_embeddings)
        
        # load control mask    
        control_mask = cv2.imread(control_mask_path)
        control_mask = cv2.cvtColor(control_mask, cv2.COLOR_BGR2RGB)
        control_mask = cv2.resize(control_mask, (512,512))
        mask_show = control_mask.copy()
        control_mask = torch.from_numpy(control_mask).permute(2,0,1).unsqueeze(0).to(torch.float32).cuda() / 255.0
        
        # adversarial grad
        controller = AttentionStore()
        start = time.time()            
        image_inv, x_t, worst_mask, vis, worst_iou = text2image_ldm_stable_last(ldm_stable, [prompt], controller, latent=x_t, num_inference_steps=NUM_DDIM_STEPS, guidance_scale=GUIDANCE_SCALE, generator=None, uncond_embeddings=uncond_embeddings, mask_control=control_mask,raw_img=raw_img,boxes=boxes, label_masks_256=label_masks_256)    
        print('Grad Time:', time.time() - start)
        
        # show 
        ptp_utils.view_images([image_inv[0]], prefix=os.path.join(save_path,'adv','sa_'+str(i)), shuffix='.jpg')
        ptp_utils.view_images([raw_img_show, mask_show, image_inv[0], worst_mask, vis, str2img(worst_iou)], prefix=os.path.join(save_path,'pair','sa_'+str(i)), shuffix='.jpg')
        
        # record adversarial iou
        with open(save_path+'/record/sa_'+str(i)+'.txt','w') as f:
            f.write(str(worst_iou))