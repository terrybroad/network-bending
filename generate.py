import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm

def create_layer_channel_dim_dict(channel_multiplier):
    layer_channel_dict = {
        0: 512,
        1: 512,
        2: 512,
        3: 512,
        4: 512,
        5: 512,
        6: 512,
        7: 256*channel_multiplier,
        8: 256*channel_multiplier,
        9: 128*channel_multiplier,
        10: 128*channel_multiplier,
        11: 64*channel_multiplier,
        12: 64*channel_multiplier,
        13: 32*channel_multiplier,
        14: 32*channel_multiplier,
        15: 16*channel_multiplier,
        16: 16*channel_multiplier
    }
    return layer_channel_dict

def generate(args, g_ema, device, mean_latent, t_dict_list):

    with torch.no_grad():
        g_ema.eval()
        for i in tqdm(range(args.pics)):
           sample_z = torch.randn(args.sample, args.latent, device=device)

           sample, _ = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list)
           
           utils.save_image(
            sample,
            f'sample/{str(i).zfill(6)}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

def create_random_layer_transform_dict(layer,transform,percentage, params):
    indicies = range(0,512)
    transform_dict ={
        "layerID": layer,
        "transformID": transform,
        "indicies": indicies,
        "params": params
    }
    print(transform_dict['params'])
    return transform_dict

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--sample', type=int, default=1)
    parser.add_argument('--pics', type=int, default=20)
    parser.add_argument('--truncation', type=float, default=1)
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

    transform_dict_list = []
    transform_dict_list.append( create_random_layer_transform_dict(5,"scalar-multiply",0.5, [10.0]) )
    generate(args, g_ema, device, mean_latent, transform_dict_list)
