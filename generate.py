import argparse
import torch
import os

from torchvision import utils
from model import Generator
from tqdm import tqdm
from util import *

def generate(args, g_ema, device, mean_latent, t_dict_list):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

            if not os.path.exists('sample'):
                os.makedirs('sample')

            utils.save_image(
                sample,
                f'sample/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1),
        )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)

    args = parser.parse_args()


    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    new_state_dict = g_ema.state_dict()
    checkpoint = torch.load(args.ckpt)
    
    ext_state_dict  = torch.load(args.ckpt)['g_ema']
    g_ema.load_state_dict(checkpoint['g_ema'])
    new_state_dict.update(ext_state_dict)
    g_ema.load_state_dict(new_state_dict)
    g_ema.eval()
    g_ema.to(device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    layer_channel_dims = create_layer_channel_dim_dict(args.channel_multiplier)
    transform_dict_list = []
    transform_dict_list.append(create_random_transform_dict(7,layer_channel_dims,"flip-h",[0.5], 0.5))
    generate(args, g_ema, device, mean_latent, transform_dict_list)
