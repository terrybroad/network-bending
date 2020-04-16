import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Function
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

from clustering_models import FeatureClassifier_L1

transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
train_data = datasets.ImageFolder('/home/terence/data/network-bending/activations/1',  transform=transform)
device = 'cuda'


data_loader = DataLoader(train_data, batch_size=100,shuffle=True)
dataset = iter(data_loader)

classifier_l1 = FeatureClassifier_L1().to(device)
criterion = nn.CrossEntropyLoss()
criterion.to(device)
classifier_l1.train()
optimizer = optim.Adam(classifier_l1.parameters(), lr=0.0001)

for step in range(100000):
    try:
        image, label = next(dataset)
    except (OSError, StopIteration):
        dataset = iter(data_loader)
        image, label = next(dataset)

    image = image.to(device)
    norm_image = ( image - 0.5 ) * 2
    label = label.to(device)
    vec, x_prob = classifier_l1(norm_image)
    loss = criterion(x_prob, label)
    loss = loss.to(device)
    print("step: "+str(step).zfill(6) +", loss: " + str(loss))
    loss.backward()
    optimizer.step()
    if step % 10000 == 0:
        torch.save(classifier_l1.state_dict(), 'classifier1_'+str(step)+'.pt')

