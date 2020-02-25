from torch import nn
from torch.nn import init
from torch.nn import functional as F
import torch.utils.cpp_extension
import random

class Erode(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, params, indicies):
        #check params
        x_array = list(torch.split(x,1,1))
        for i, dim in enumerate(x_array):
            if i in indicies:
                d_ = torch.squeeze(dim)
                feat = torch.ops.my_ops.erode(d_,params[0])
                feat = torch.unsqueeze(torch.unsqueeze(feat,0),0)
                x_array[i] = feat
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
            
