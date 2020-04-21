import torch
import collections
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

layer_depth_dict = {
    1 : 0,
    2 : 0,
    3 : 1,
    4 : 1,
    5 : 2,
    6 : 2,
    7 : 3,
    8 : 3,
    9 : 4,
    10 : 4,
    11 : 5,
    12 : 5,
    13 : 6,
    14 : 6,
    15 : 7,
    16 : 7
}

layer_class_dict = {
    1 : 512,
    2 : 512,
    3 : 512,
    4 : 512,
    5 : 512,
    6 : 512,
    7 : 512,
    8 : 512,
    9 : 256,
    10 : 256,
    11 : 128,
    12 : 128,
    13 : 64,
    14 : 64,
    15 : 32,
    16 : 32
}

#based on https://github.com/pytorch/vision/blob/master/torchvision/models/shufflenetv2.py
def channel_shuffle(x, groups):
    # type: (torch.Tensor, int) -> torch.Tensor
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride):
        super(InvertedResidual, self).__init__()

        if not (1 <= stride <= 3):
            raise ValueError('illegal stride value')
        self.stride = stride

        branch_features = oup // 2
        assert (self.stride != 1) or (inp == branch_features << 1)

        if self.stride > 1:
            self.branch1 = nn.Sequential(
                self.depthwise_conv(inp, inp, kernel_size=3, stride=self.stride, padding=1),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(branch_features),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch1 = nn.Sequential()

        self.branch2 = nn.Sequential(
            nn.Conv2d(inp if (self.stride > 1) else branch_features,
                      branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
            self.depthwise_conv(branch_features, branch_features, kernel_size=3, stride=self.stride, padding=1),
            nn.BatchNorm2d(branch_features),
            nn.Conv2d(branch_features, branch_features, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(branch_features),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def depthwise_conv(i, o, kernel_size, stride=1, padding=0, bias=False):
        return nn.Conv2d(i, o, kernel_size, stride, padding, bias=bias, groups=i)

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)

        out = channel_shuffle(out, 2)

        return out

class FeatureClassifier(nn.Module):
    def __init__(self, layer_depth, bottleneck):
        super().__init__()

        self.num_layers = layer_depth_dict[layer_depth]
        self.bottleneck = bottleneck
        input_channels = 1
        output_channels = 50
        classes = layer_class_dict[layer_depth]
        self.first_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        shuffle_layers_dict = collections.OrderedDict()
        for i in range(self.num_layers):
            shuffle_layers_dict[str(i) + "res"] = InvertedResidual(output_channels, output_channels, 1)
            shuffle_layers_dict[str(i) + "maxpool"] = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            print(i)
        
        self.shuffleLayers = nn.Sequential(shuffle_layers_dict)
        
        self.final_conv = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )
        self.fc_bottleneck = nn.Linear(output_channels, self.bottleneck)
        self.fc_out = nn.Linear(self.bottleneck, classes)


    def forward(self, x):
        x = self.first_conv(x)
        x = self.maxpool(x)
        if self.num_layers > 0:
            x = self.shuffleLayers(x)
        x = x.mean([2,3])
        feat_vec = self.fc_bottleneck(x)
        class_probs = self.fc_out(feat_vec)
        return feat_vec, class_probs