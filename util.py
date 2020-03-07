import random

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

def create_random_transform_dict(layer, layer_channel_dict, transform, params, percentage):
    layer_dim = layer_channel_dict[layer]
    num_samples = int( layer_dim * percentage )
    rand_indicies = random.sample(range(0, layer_dim), num_samples)
    transform_dict ={
        "layerID": layer,
        "transformID": transform,
        "indicies": rand_indicies,
        "params": params
    }
    return transform_dict

def create_layer_wide_transform_dict(layer, layer_channel_dict, transform, params):
    layer_dim = layer_channel_dict[layer]
    transform_dict ={
        "layerID": layer,
        "transformID": transform,
        "indicies": range(0, layer_dim),
        "params": params
    }
    return transform_dict