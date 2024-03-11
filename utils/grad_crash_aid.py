import os
import argparse
import sys
############## Initialize #####################
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# sam setting
parser.add_argument('--model', type=str, default='sam', help='cnn')
parser.add_argument('--model_type', type=str, default='vit_b', help='cnn')
parser.add_argument('--sam_batch', type=int, default=150, help='cnn')

# SD setting
parser.add_argument('--ddim_steps', default=50, type=int, help='random seed')
parser.add_argument('--guess_mode', action='store_true')   
parser.add_argument('--guidance_scale', default=7.5, type=float, help='random seed') 
<<<<<<< HEAD
=======
parser.add_argument('--random_latent', action='store_true')
>>>>>>> e2255d478443aaa0c3130e5940861215d9b642c2

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
parser.add_argument('--data_root', default='/data/tanglv/data/sam-1b/sa_000000', type=str, help='random seed')   
parser.add_argument('--save_root', default='output/sa_000000-Grad', type=str, help='random seed')   
parser.add_argument('--control_mask_dir', default='/data/tanglv/data/sam-1b/sa_000000', type=str, help='random seed')   
parser.add_argument('--inversion_dir', default='output/sa_000000-Inversion/embeddings', type=str, help='random seed')   
parser.add_argument('--caption_path', default='/data/tanglv/data/sam-1b/sa_000000-blip2-caption.json', type=str, help='random seed')    
parser.add_argument('--controlnet_path', default='ckpt/control_v11p_sd15_mask_sa000000.pth', type=str, help='random seed')    

args = parser.parse_args()
print("Check Status")


if __name__ == '__main__':
    # Prepare save path
    save_path = args.save_root + '/' + args.prefix + '-SD-' + str(args.guidance_scale) + '-' +str(args.ddim_steps) +'-SAM-' + args.model + '-' + args.model_type +'-'+ str(args.sam_batch)+ '-ADV-' + str(args.eps) + '-' +str(args.steps)  + '-' + str(args.alpha) + '-' + str(args.mu)+  '-' +  str(args.kappa) +'-'+ str(args.gamma) + '-' + str(args.beta) + '-' + str(args.norm) 
<<<<<<< HEAD

=======
    if args.random_latent:
        save_path += '-random_latent'
>>>>>>> e2255d478443aaa0c3130e5940861215d9b642c2
    # Adversarial grad loop
    for i in range(args.start, args.end+1):
        if os.path.exists(os.path.join(args.inversion_dir,'sa_'+str(i)+'_latent.pth')) and \
        not os.path.exists(os.path.join(save_path, 'adv', 'sa_'+str(i)+'.png')):
            sys.exit(-1)
    sys.exit(0)