import argparse
import torch
import yaml
import os

from torchvision import utils
from model import Generator
from tqdm import tqdm
from util import *

def create_transform_dict_list(layer, layer_channel_dict, transform):
    transform_dict_list = []
    if transform['features'] == 'all':
            transform_dict_list.append(
                create_layer_wide_transform_dict(layer,
                    layer_channel_dict, 
                    transform['transform'], 
                    transform['params']))
    elif transform['features'] == 'random':
            transform_dict_list.append(
                create_random_transform_dict(layer,
                    layer_channel_dict, 
                    transform['transform'], 
                    transform['params'],
                    transform['feature-param']))
    else:
            print('transform type: ' + str(transform) + ' not recognised')
    return transform_dict_list

def generate_strips(args, g_ema, device, mean_latent, layer_channel_dims, config):
    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)

            if not os.path.exists('sample'):
                os.makedirs('sample')
            
            images = []
            for key in layer_channel_dims:
                t_dict_list = create_transform_dict_list(key, layer_channel_dims, config)
                sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)
                images.append(sample)

            image = torch.cat(images)
            utils.save_image(
                image,
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
    parser.add_argument('--config', type=str, default="sample_strip_config.yaml")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    yaml_config = {}
    with open(args.config, 'r') as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
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
    generate_strips(args, g_ema, device, mean_latent, layer_channel_dims, yaml_config)
    
    with open(r'sample/config.yaml', 'w') as file:
        documents = yaml.dump(yaml_config, file)

