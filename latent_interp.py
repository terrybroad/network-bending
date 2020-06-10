import argparse
import math
import torch
import yaml
import os
import random

from torchvision import utils
from model import Generator
from tqdm import tqdm
from util import *
import numpy as np

def get_noise_list(size):
    log_size = int(math.log(size, 2))
    noise_list = [[1, 1, 2 ** 2, 2 ** 2]]
    for i in range(3, log_size + 1):
        for j in range(2):
                noise_list.append([1, 1, 2 ** i, 2 ** i])
    return noise_list


def slerp(val, low, high):
    '''
    original: Animating Rotation with Quaternion Curves, Ken Shoemake
    
    https://arxiv.org/abs/1609.04468
    Code: https://github.com/soumith/dcgan.torch/issues/14, Tom White
    '''
    omega = np.arccos(np.dot(low/np.linalg.norm(low), high/np.linalg.norm(high)))
    so = np.sin(omega)
    return np.sin((1.0-val)*omega) / so * low + np.sin(val*omega)/so * high


def get_slerp_interp(nb_latents, nb_interp):
    low = np.random.randn(512)
    latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
    for _ in range(nb_latents):
            high = np.random.randn(512)#low + np.random.randn(512) * 0.7

            interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
            latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                            dtype=np.float32)

            latent_interps = np.vstack((latent_interps, latent_interp))
            low = high

    return latent_interps

def get_slerp_loop(nb_latents, nb_interp):
        low = np.random.randn(512)
        og_low = low
        latent_interps = np.empty(shape=(0, 512), dtype=np.float32)
        for _ in range(nb_latents):
                high = np.random.randn(512)#low + np.random.randn(512) * 0.7

                interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
                latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                                dtype=np.float32)

                latent_interps = np.vstack((latent_interps, latent_interp))
                low = high
        
        interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, og_low) for v in interp_vals],
                                                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        return latent_interps

def get_slerp_loop_noise(nb_latents, nb_interp, shape):
        low = np.random.randn(shape[0],shape[1],shape[2],shape[3])
        og_low = low
        latent_interps = np.empty(shape=(shape[0],shape[1],shape[2],shape[3]), dtype=np.float32)
        for _ in range(nb_latents):
                high = np.random.randn(shape[0],shape[1],shape[2],shape[3])#low + np.random.randn(512) * 0.7

                interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
                latent_interp = np.array([slerp(v, low, high) for v in interp_vals],
                                                dtype=np.float32)

                latent_interps = np.vstack((latent_interps, latent_interp))
                low = high
        
        interp_vals = np.linspace(1./nb_interp, 1, num=nb_interp)
        latent_interp = np.array([slerp(v, low, og_low) for v in interp_vals],
                                                dtype=np.float32)
        latent_interps = np.vstack((latent_interps, latent_interp))
        return latent_interps

#1 min slerps = get_slerp_loop(32, 45)
def interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
    slerps = get_slerp_loop(32, 45)
    # noise_slerps = []
    # noise_shape_list = get_noise_list(args.size)
    # for shape in noise_shape_list:
    #     noise_slerps.append(get_slerp_loop_noise(32, 35, shape)) 
    t_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)
    for i in range(len(slerps)):
        print('generating frame: ' + str(i))
        input = torch.tensor(slerps[i])
        input = input.view(1,512)
        input = input.to(device)
        # noises = []
        # for layer_n in noise_slerps:
        #     noises.append(torch.tensor(layer_n[i].to(device)))
        image, _ = g_ema([input],truncation=args.truncation, randomize_noise=False, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

        if not os.path.exists('interp'):
            os.makedirs('interp')
        
        utils.save_image( 
                    image,
                    'interp/'+str(i + 1).zfill(6)+'.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                    padding=0)

def multiple_transform_interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims):
    slerps = get_slerp_loop(32, 45)
    
    transform_dict_list_list = create_transforms_dict_list_list(yaml_config,cluster_config, layer_channel_dims)
    t_index = 0
    for t_dict_list in transform_dict_list_list:
        for i in range(len(slerps)):
            print('generating frame: ' + str(i))
            input = torch.tensor(slerps[i])
            input = input.view(1,512)
            input = input.to(device)
            image, _ = g_ema([input],truncation=args.truncation, randomize_noise=False, truncation_latent=mean_latent, transform_dict_list=t_dict_list)

            if not os.path.exists('interp/'+str(t_index)+'/'):
                os.makedirs('interp/'+str(t_index)+'/')
            
            utils.save_image( 
                        image,
                        'interp/'+str(t_index)+'/'+str(i + 1).zfill(6)+'.png',
                        nrow=1,
                        normalize=True,
                        range=(-1, 1),
                        padding=0)
        t_index += 1


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
    parser.add_argument('--load_clusters', type=str, default="configs/example_cluster_dict.yaml")
    parser.add_argument('--multiple_transforms',type=int, default=0)

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
    if args.load_clusters != "":
        with open(args.load_clusters, 'r') as stream:
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
    
    if args.multiple_transforms == 1:
        transform_dict_list = create_transforms_dict_list_list(yaml_config, cluster_config, layer_channel_dims)
    else:
        transform_dict_list = create_transforms_dict_list(yaml_config, cluster_config, layer_channel_dims)

    if args.load_latent == "":
        if args.multiple_transforms == 1:
            multiple_transform_interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
        else:
            interp(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims)
    
    # else:
        # Do nothing for now
        # latent=torch.load(args.load_latent)['latent']
        # noises=torch.load(args.load_latent)['noises']
        # generate_from_latent(args, g_ema, device, mean_latent, yaml_config, cluster_config, layer_channel_dims, latent, noises)
    
    # config_out = {}
    # config_out['transforms'] = yaml_config['transforms']
    # with open(r'sample/config.yaml', 'w') as file:
    #     documents = yaml.dump(config_out, file)

    #generator.style.to('cuda')
    # with torch.no_grad():
    #     interp(generator, step, mean_style)



