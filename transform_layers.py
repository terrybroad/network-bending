from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.utils.cpp_extension
import random

torch.ops.load_library("transforms/erode/build/liberode.so")
torch.ops.load_library("transforms/dilate/build/libdilate.so")
torch.ops.load_library("transforms/scale/build/libscale.so")
torch.ops.load_library("transforms/rotate/build/librotate.so")
torch.ops.load_library("transforms/translate/build/libtranslate.so")

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], int) or params[0] < 0):
            print('Erosion parameter must be a positive integer')
            raise ValueError
        
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.erode(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Dilate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], int) or params[0] < 0):
            print('Dilation parameter must be a positive integer')
            raise ValueError
        
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.dilate(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Translate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], float) or not isInstance(params[1], float)
         or params[0] < -1 or params[0] > 1 or params[1] < -1 or params[1] > 1):
            print('Translation must have two parameters, which should be floats between -1 and 1.')
            raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.translate(d_,params[0], params[1])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Scale(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], float)):
            print('Scale parameter should be a float.')
            raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.scale(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Rotate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], float) or params[0] < 0 or params[0] > 360):
            print('Rotation parameter should be a float between 0 and 360 degrees.')
            raise ValueError
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = torch.ops.my_ops.rotate(d_,params[0])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class FlipHorizontal(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                tf = d_.flip([3])
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Invert(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                ones = torch.ones(d_.size(), dtype=d_.dtype, layout=d_.layout, device=d_.device)
                tf = ones - d_
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class BinaryThreshold(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], float) or params[0] < -1 or params[0] > 1):
            print('Binary threshold parameter should be a float between -1 and 1.')
            raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = Variable(torch.Tensor([params[0]]))
                tf = (d_ > t).float() * 1
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ScalarMultiply(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        if(not isInstance(params[0], float)):
            print('Scalar multiply parameter should be a float between -1 and 1.')
            raise ValueError

        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = d_ * param[0]
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class Ablate(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                t = d_ * 0
                tf = torch.unsqueeze(torch.unsqueeze(tf,0),0)
                x_array[i] = tf
        return torch.cat(x_array,1)

class ManipulationLayer(nn.Module):
    def __init__(self, layerID):
        super().__init__()
        self.layerID = layerID
        layer_options = {
            "erode" : self.erode
        }
        self.erode = Erode()

    def forward(self, input, tranforms_dict):
        out = input
        for layerID, transformID, params, indicies in tranforms_dict:
            if layerID == self.layerID:
                out = self.layer_options[transformID](out, params, indicies)
        return out
            
