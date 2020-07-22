import argparse
import torch
import yaml
import os

from torchvision import utils
from model import Generator
from tqdm import tqdm
from util import *

def generate(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
    with torch.no_grad():
        g_ema.eval()
        t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
        for i in tqdm(range(args.pics)):
            sample_z = torch.randn(args.sample, args.latent, device=device)
            print(sample_z.size())
            sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

            if not os.path.exists('sample'):
                os.makedirs('sample')

            utils.save_image(
                sample,
                f'sample/{str(i).zfill(6)}.png',
                nrow=1,
                normalize=True,
                range=(-1, 1))

def generate_from_latent(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noise):
    with torch.no_grad():
        g_ema.eval()
        slice_latent = latent[0,:]
        slce_latent = slice_latent.unsqueeze(0)
        print(slice_latent.size())
        for i in tqdm(range(args.pics)):
            t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
            sample, _ = g_ema([slce_latent], input_is_latent=True, noise=noises, transform_dict_list=t_dict_list)

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
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--config', type=str, default="configs/example_transform_config.yaml")
    parser.add_argument('--load_latent', type=str, default="") 
    parser.add_argument('--clusters', type=str, default="configs/example_cluster_dict.yaml")

    args = parser.parse_args()

    args.latent = 512
    args.n_mlp = 8

    yaml_config = {}
    with open(args.config, 'r') as stream:
        try:
            yaml_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    
    cluster_config = {}
    if args.clusters != "":
        with open(args.clusters, 'r') as stream:
            try:
                cluster_config = yaml.load(stream)
            except yaml.YAMLError as exc:
                print(exc)

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    new_state_dict = g_ema.state_dict()
    checkpoint = torch.load(args.ckpt)
    
    ext_state_dict  = torch.load(args.ckpt)['g_ema']
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
    transform_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
    
    if args.load_latent == "":
        generate(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
    else:
        latent=torch.load(args.load_latent)['latent']
        noises=torch.load(args.load_latent)['noises']
        generate_from_latent(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noises)
    
    config_out = {}
    config_out['transforms'] = yaml_config['transforms']
    with open(r'sample/config.yaml', 'w') as file:
        documents = yaml.dump(config_out, file)

