import os
import torch
import argparse
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Function
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tensorboardX import SummaryWriter

from clustering_models import FeatureClassifier


def train_classifier(layer, batch_size, n_epochs, bottleneck, data_str, save_str):
    transform = transforms.Compose([transforms.Grayscale(),transforms.ToTensor()])
    dataset = datasets.ImageFolder(data_str +'/'+str(layer),  transform=transform)
    device = 'cuda'
    writer = SummaryWriter()
    validation_split = 0.1
    dataset_len = len(dataset)
    indices = list(range(dataset_len))
    data_save_root = save_str+'/'+str(layer)+"/"
    if not os.path.exists(data_save_root):
                os.makedirs(data_save_root)

    # Randomly splitting indices:
    val_len = int(np.floor(validation_split * dataset_len))
    validation_idx = np.random.choice(indices, size=val_len, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    ## Defining the samplers for each phase based on the random indices:
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
    validation_loader = torch.utils.data.DataLoader(dataset, sampler=validation_sampler, batch_size=batch_size)
    data_loaders = {"train": train_loader, "valid": validation_loader}
    data_lengths = {"train": len(train_idx), "valid": val_len}

    classifier = FeatureClassifier(layer,bottleneck).to(device)
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    lr = 0.0001
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)

    hparam_dict = {
        "Layer" : layer,
        "batch size" : batch_size,
        "Learning rate": lr
    }
    optimizer.zero_grad()
    writer.add_hparams(hparam_dict, {})
    total_it = 0
    for epoch in range(n_epochs):
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                classifier.train(True)  # Set model to training mode
            else:
                optimizer.zero_grad()
            running_loss = 0.0
            epoch_it = 0
            for image, label in data_loaders[phase]:
                classifier.zero_grad()
                optimizer.zero_grad()
                image = image.to(device)
                norm_image = ( image - 0.5 ) * 2
                label = label.to(device)
                vec, x_prob = classifier(norm_image)
                loss = criterion(x_prob, label)
                loss = loss.to(device)
                running_loss += loss.detach()
                if phase == 'train':
                    print("layer: " +str(layer)+ ", epoch: " +str(epoch)+ ", step: "+str(epoch_it).zfill(6) +", training loss: " + str(float(loss)))
                    writer.add_scalar('data/train_loss_continous', loss, total_it)
                    loss.backward()
                    optimizer.step()
                    total_it +=1                # optimizer = scheduler(optimizer, epoch)
                epoch_it +=1
            
            epoch_loss = running_loss / data_lengths[phase]
            if phase == 'train':    
                print("Epoch: "+str(epoch).zfill(6) +", train loss: " + str(epoch_loss))
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
            if phase == 'valid':
                print("Epoch: "+str(epoch).zfill(6) +", valid loss: " + str(epoch_loss))
                writer.add_scalar('data/valid_loss_epoch', epoch_loss, epoch)

    if epoch % 10 == 0:
        torch.save(classifier.state_dict(), data_save_root+'/classifier'+str(layer)+'_'+str(epoch)+'.pt')    

    torch.save(classifier.state_dict(), data_save_root +'/classifier'+str(layer)+'_final.pt')  
    writer.export_scalars_to_json(data_save_root+"all_scalars.json")
    writer.close()   

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--bottleneck', type=int, default=10)
    parser.add_argument('--data', type=str, default='activations/')
    parser.add_argument('--save',type=str, default='models/classifiers')
    parser.add_argument('--first_layer', type=int, default=1)
    parser.add_argument('--last_layer', type=int, default=8)
    args = parser.parse_args()

    for i in range(args.first_layer, args.last_layer):
        train_classifier(layer=i, batch_size=args.batch_size, n_epochs=args.n_epochs, bottleneck=args.bottleneck, data_str=args.data, save_str=args.save)