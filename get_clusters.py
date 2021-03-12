import argparse
import torch
import yaml
import os
import copy

from torchvision import utils
from model import Generator
from clustering_models import FeatureClassifier
from tqdm import tqdm
from util import *
from kmeans_pytorch import kmeans

cluster_layer_dict = {
    1 : 5,
    2 : 5,
    3 : 5,
    4 : 5,
    5 : 5,
    6 : 5,
    7 : 5,
    8 : 5,
    9 : 4,
    10 : 4,
    11 : 4,
    12 : 4,
    13 : 3,
    14 : 3,
    15 : 3,
    16 : 3
}

def get_clusters_from_generated_greedy(args, g_ema, device, mean_latent, t_dict_list, yaml_config, layer_channel_dims):
    print("get clusters")
    with torch.no_grad():
        g_ema.eval()

        latent_ll = []
        feature_ll = []
        feature_cluster_sum_dict = {}
        feature_cluster_dict = {}
        
        for i in tqdm(range(args.n_layers)):
            true_index = i+1
            latent_list = []
            feature_list = []
            latent_ll.append(latent_list)
            feature_ll.append(feature_list)
            feature_cluster_sum_dict[true_index] = {}
            for j in tqdm(range(layer_channel_dims[true_index])):
                feature_cluster_sum_dict[true_index][j] = [] 
        
        for i in tqdm(range(args.num_samples)):
            print("processing sample: " + str(i))
            sample_z = torch.randn(1, args.latent, device=device) 
            sample, activation_maps = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list, return_activation_maps=True)
            for index, activations in enumerate(activation_maps):
                true_index = index+1
                classifier = FeatureClassifier(true_index)
                classifier_str = args.classifier_ckpts + "/" + str(true_index) + "/classifier" + str(true_index) + "_final.pt"
                classifier_state_dict = torch.load(classifier_str)
                classifier.load_state_dict(classifier_state_dict)
                classifier.to(device)
                layer_activation_maps = activation_maps[index]
                a_map_array = list(torch.split(layer_activation_maps,1,1))
                for j, map in enumerate(a_map_array):
                    map = map.to(device)
                    feat_vec, class_prob = classifier(map)
                    latent_ll[index].append(feat_vec)
                    feature_ll[index].append(j)
        
        for i in tqdm(range(args.n_layers)):
            true_index = i+1
            print("generating clusters for layer: " + str(i))
            cluster_ids_x, cluster_centers = kmeans(X=torch.stack(latent_ll[i]), num_clusters=cluster_layer_dict[true_index], distance='euclidean', device=torch.device('cuda'))
            for j, id in enumerate(cluster_ids_x):
                feature_cluster_sum_dict[true_index][feature_ll[i][j]].append(id)
            
            dict_list = []
            for j in tqdm(range(layer_channel_dims[true_index])):
                cluster_id = max(feature_cluster_sum_dict[true_index][j])
                cluster_dict = {"feature_index": int(j), "cluster_index": int(cluster_id)}
                dict_list.append(cluster_dict)
            feature_cluster_dict[true_index] = dict_list
    
    with open(r'cluster_dict.yaml', 'w') as file:
        documents = yaml.dump(feature_cluster_dict, file)

def get_clusters_from_generated_average(args, g_ema, device, mean_latent, t_dict_list, yaml_config, layer_channel_dims):
    print("get clusters")
    with torch.no_grad():
        g_ema.eval()

        latent_ll = []
        feature_ll = []
        feature_cluster_sum_dict = {}
        feature_cluster_dict = {}
        feature_latent_dict = {}
        
        for i in tqdm(range(args.n_layers)):
            print("I" + str(i))
            true_index = i+1
            latent_list = []
            feature_list = []
            latent_ll.append(latent_list)
            feature_ll.append(feature_list)
            feature_cluster_sum_dict[true_index] = {}
            for j in tqdm(range(layer_channel_dims[true_index])):
                feature_cluster_sum_dict[true_index][j] = 0 
                latent_ll[i].append(0)

        for i in tqdm(range(args.num_samples)):
            print("processing sample: " + str(i))
            sample_z = torch.randn(1, args.latent, device=device) 
            sample, activation_maps = g_ema([sample_z], truncation=args.truncation, truncation_latent=mean_latent, transform_dict_list=t_dict_list, return_activation_maps=True)
            for index, activations in enumerate(activation_maps):
                if index < args.n_layers:
                    true_index = index+1
                    classifier = FeatureClassifier(true_index)
                    classifier_str = args.classifier_ckpts + "/" + str(true_index) + "/classifier" + str(true_index) + "_final.pt"
                    classifier_state_dict = torch.load(classifier_str)
                    classifier.load_state_dict(classifier_state_dict)
                    classifier.to(device)
                    layer_activation_maps = activation_maps[index]
                    a_map_array = list(torch.split(layer_activation_maps,1,1))
                    for j, map in enumerate(a_map_array):                  
                        map = map.to(device)
                        feat_vec, class_prob = classifier(map)
                        normalised_feat_vec = feat_vec / args.num_samples
                        latent_ll[index][j] = latent_ll[index][j] + normalised_feat_vec
                        # feature_ll[index].append(j)
        
        for i in tqdm(range(args.n_layers)):
            true_index = i+1
            print("generating clusters for layer: " + str(i))
            cluster_ids_x, cluster_centers = kmeans(X=torch.stack(latent_ll[i]), num_clusters=cluster_layer_dict[true_index], distance='euclidean', device=torch.device('cuda'))
            dict_list = []
            latent_dict_list = []
            for j, id in enumerate(cluster_ids_x):
                cluster_dict = {"feature_index": int(j), "cluster_index": int(id)}
                latent_dict = {"feature_index": int(j), "latent": latent_ll[i][j].to('cpu').numpy().tolist()}
                dict_list.append(cluster_dict)
                latent_dict_list.append(latent_dict)
            feature_cluster_dict[true_index] = dict_list
            feature_latent_dict[true_index] = latent_dict_list
    
    with open(r'cluster_dict.yaml', 'w') as file:
        documents = yaml.dump(feature_cluster_dict, file)
    with open(r'latent_dict.yaml', 'w') as file:
        documents = yaml.dump(feature_latent_dict, file)


def get_clusters_from_latent(args, g_ema, device, mean_latent, t_dict_list, yaml_config, layer_channel_dims, latent, noise):
    print("get clusters")
    with torch.no_grad():
        g_ema.eval()
        slice_latent = latent[0,:]
        slce_latent = slice_latent.unsqueeze(0)
        print(slice_latent.size())
        sample, activation_maps = g_ema([slce_latent], input_is_latent=True, noise=noises, transform_dict_list=t_dict_list, return_activation_maps=True)
        print(len(activation_maps))
        feature_cluster_dict = {}
        for index, activations in enumerate(activation_maps):
                if index < args.n_layers:
                    true_index = index+1
                    classifier = FeatureClassifier(true_index)
                    classifier_str = args.classifier_ckpts + "/" + str(true_index) + "/classifier" + str(true_index) + "_final.pt"
                    classifier_state_dict = torch.load(classifier_str)
                    classifier.load_state_dict(classifier_state_dict)
                    classifier.to(device)
                    layer_activation_maps = activation_maps[index]
                    a_map_array = list(torch.split(layer_activation_maps,1,1))
                    dict_list = []
                    latent_list = []
                    for i, map in enumerate(a_map_array):
                        map = map.to(device)
                        feat_vec, class_prob = classifier(map)
                        activation_dict = {"class_index": i, "feat_vec": feat_vec}
                        # dict_list.append(activation_dict)
                        latent_list.append(feat_vec)
                    cluster_ids_x, cluster_centers = kmeans(X=torch.stack(latent_list), num_clusters=cluster_layer_dict[true_index], distance='euclidean', device=torch.device('cuda'))
                    for i, id in enumerate(cluster_ids_x):
                        cluster_dict = {"feature_index": int(i), "cluster_index": int(id)}
                        dict_list.append(cluster_dict)
                    feature_cluster_dict[true_index] = dict_list
        with open(r'cluster_dict.yaml', 'w') as file:
            documents = yaml.dump(feature_cluster_dict, file)
            





if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--truncation', type=float, default=0.5)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--ckpt', type=str, default="models/stylegan2-ffhq-config-f.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--config', type=str, default="configs/empty_transform_config.yaml")
    parser.add_argument('--load_latent', type=str, default="") 
    parser.add_argument('--classifier_ckpts', type=str, default="models/classifiers")
    parser.add_argument('--n_layers', type=int, default=12)

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
    new_state_dict.update(ext_state_dict)
    g_ema.load_state_dict(new_state_dict)
    g_ema.eval()
    g_ema.to(device)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None
    
    layer_channel_dims = create_layer_channel_dim_dict(args.channel_multiplier, args.n_layers)
    transform_dict_list = create_transforms_dict_list(yaml_config, {}, layer_channel_dims)
    
    if args.load_latent == "":
        get_clusters_from_generated_average(args, g_ema, device, mean_latent, transform_dict_list, yaml_config, layer_channel_dims)
    else:
        latent=torch.load(args.load_latent)['latent']
        noises=torch.load(args.load_latent)['noises']
        get_clusters_from_latent(args, g_ema, device, mean_latent, transform_dict_list, yaml_config, layer_channel_dims, latent, noises)
    
