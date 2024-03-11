# from swinT import get_swin_B
# from pytorch_pretrained_vit import ViT, load_pretrained_weights
import torchvision
import torch
import os
import timm
# from resnet_adv import resnet152_denoise, resnet101_denoise, resnet152
# from aux_bn import MixBatchNorm2d, to_mix_status, to_clean_status, to_adv_status
# from efficientnet_pytorch import EfficientNet
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from sam_continue_learning.segment_anything_training import sam_model_registry
import os
from sam_continue_learning.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

def get_model(model, model_type):
    home_path = 'home_path'
    if model == 'sam':
        if model_type == "vit_b": sam_checkpoint = "sam_continue_learning/pretrained_checkpoint/sam_vit_b_01ec64.pth"
        if model_type == "vit_l": sam_checkpoint = "sam_continue_learning/pretrained_checkpoint/sam_vit_l_0b3195.pth"
        if model_type == "vit_h": sam_checkpoint = "sam_continue_learning/pretrained_checkpoint/sam_vit_h_4b8939.pth"
        
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        return sam
    elif model == 'sam_efficient':
        
        if model_type == "vit_t": sam_checkpoint = torch.load("sam_continue_learning/pretrained_checkpoint/efficient_sam_vitt.pt")['model']
        if model_type == "vit_s": sam_checkpoint = torch.load("sam_continue_learning/pretrained_checkpoint/efficient_sam_vits.pt")['model']
        
        if model_type == 'vit_t': sam = build_efficient_sam_vitt()
        if model_type == 'vit_s': sam = build_efficient_sam_vits()      
        sam.load_state_dict(sam_checkpoint)
        # print(type(sam))
        # raise NameError
        return sam
    elif model == 'resnet50':
        net = torchvision.models.resnet50(pretrained=True)
        #net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/resnet50-0676ba61.pth')))
    elif model == 'mnv2':
        net = torchvision.models.mobilenet_v2()
        net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/mobilenet_v2-b0353104.pth')))
    elif model == 'vgg19':
        # https://download.pytorch.org/models/vgg19-dcbb9e9d.pth
        net = torchvision.models.vgg19()
        net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/vgg19-dcbb9e9d.pth')))
    elif model == 'vit':
        # https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz
        net = ViT('B_16_imagenet1k_224', pretrained=False)
        load_pretrained_weights(net, weights_path=os.path.join(home_path, 'checkpoints/B_16_imagenet1k_224.pth'))
    elif model == 'swint':
        net = get_swin_B()
    elif model == 'inception_v3':
        net = torchvision.models.inception_v3(pretrained=True)
    elif model == 'pvtv2':
        net = timm.create_model('pvt_v2_b2_li', pretrained=True, num_classes=1000)
    elif model == 'mvit':
        net = timm.create_model('mobilevit_s', pretrained=False, num_classes=1000)
        net.load_state_dict(torch.load('mobilevit_s-38a5a959.pth'))
    elif model == 'adv_resnet152_denoise':
        net = resnet152_denoise()
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/Adv_Denoise_Resnet152.pytorch'))
        net.load_state_dict(loaded_state_dict, strict=True)
    elif model == 'adv_resnext101_denoise':
        net = resnet101_denoise()
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/Adv_Denoise_Resnext101.pytorch'))
        net.load_state_dict(loaded_state_dict, strict=True)
    elif model == 'adv_resnet152':
        net = resnet152()
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/Adv_Resnet152.pytorch'))
        net.load_state_dict(loaded_state_dict, strict=True)
    elif model == 'resnet152_debiased':
        # https://livejohnshopkins-my.sharepoint.com/personal/yli286_jh_edu/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fyli286%5Fjh%5Fedu%2FDocuments%2Fdata%2FDebiasedModels%2Fres50%2Ddebiased%2Epth%2Etar
        net = torchvision.models.resnet152(norm_layer=MixBatchNorm2d)
        net = torch.nn.DataParallel(net)
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/res152-debiased.pth.tar'))
        net.load_state_dict(loaded_state_dict['state_dict'], strict=True)
    elif model == 'resnet50_trained_on_SIN_and_IN_then_finetuned_on_IN':
        net = torchvision.models.resnet50()
        net = torch.nn.DataParallel(net)
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'))
        net.load_state_dict(loaded_state_dict['state_dict'], strict=True)
    elif model == 'densenet161':
        # https://download.pytorch.org/models/densenet161-8d451a50.pth
        net = torchvision.models.densenet161()
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(os.path.join(home_path, 'checkpoints/densenet161-8d451a50.pth'))
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        net.load_state_dict(state_dict) 
    elif model == 'resnet152':
        # https://download.pytorch.org/models/resnet152-394f9c45.pth
        net = torchvision.models.resnet152()
        net.load_state_dict(torch.load(os.path.join(home_path, 'checkpoints/resnet152-394f9c45.pth')))
    elif model == 'ef_b7':
        net = EfficientNet.from_name('efficientnet-b7')
        loaded_state_dict = torch.load(os.path.join(home_path, 'checkpoints/efficientnet-b7-dcc49843.pth'))
        net.load_state_dict(loaded_state_dict, strict=True)
    return net