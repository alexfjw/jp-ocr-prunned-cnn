import torch
import torch.optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
import time
import os, sys
import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from src.datasets import Etl2Dataset
from utils.model_selection import train_test_split_indices
import argparse

etl2_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # duplicate the channels
        transforms.ToTensor(),
        transforms.Normalize((Etl2Dataset.mean, Etl2Dataset.mean, Etl2Dataset.mean),
                             (Etl2Dataset.std, Etl2Dataset.std, Etl2Dataset.std))
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # duplicate the channels
        transforms.ToTensor(),
        transforms.Normalize((Etl2Dataset.mean, Etl2Dataset.mean, Etl2Dataset.mean),
                             (Etl2Dataset.std, Etl2Dataset.std, Etl2Dataset.std))
    ])
}

def get_etl2_dataloaders():
    """
    returns train & test dataloaders for etl2 dataset
    """
    etl2 = Etl2Dataset(train_transforms=etl2_transforms['train'],
                       test_transforms=etl2_transforms['test'])
    train_indices, test_indices = train_test_split_indices(etl2)

    train_dataloader = DataLoader(etl2,
                                  batch_size=32,
                                  sampler=SubsetRandomSampler(train_indices),
                                  num_workers=2)
    test_dataloader = DataLoader(etl2,
                                  batch_size=32,
                                  sampler=SubsetRandomSampler(test_indices),
                                  num_workers=2)

    return train_dataloader, test_dataloader

# def train_vgg18_etl2():


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", dest="train")
    parser.add_argument("--prune", dest="prune")
    parser.add_argument("--train_path", type = str, default = "train")
    parser.add_argument("--test_path", type = str, default = "prune")
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)

    return parser.parse_args()

def main():
    args = get_args()

    if args.train:
        model = ...
        torch.save(model, 'vgg18_etl2')
    elif args.prune:
        model = None #TODO


if __name__ == '__main__':
    sys.exit(main())
