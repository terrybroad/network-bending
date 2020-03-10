import argparse
import torch
import yaml
import os
import copy
from PIL import Image

from torchvision import utils

image = Image.open('/home/terence/repos/network-bending/test_activations/16/00000_006.png')

# convert to grayscale tensor
# test x = x - 0.5
# test x = x * 2
# test save image normalize