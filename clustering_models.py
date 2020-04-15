import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

class FeatureClassifier_L1(nn.Module):
    def __init__():
        super().__init__()

        input_channels = 1
        output_channels = 50
        bottleneck = 10
        classes = 512
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
        )

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.fc_bottleneck = nn.Linear(output_channels, bottleneck)
        self.fc_out = nn.Linear(bottleneck, classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = x.mean([2,3])
        feat_vec = self.fc_bottleneck(x)
        class_probs = self.fc_out(feat_vec)
        return feat_vec, class_probs