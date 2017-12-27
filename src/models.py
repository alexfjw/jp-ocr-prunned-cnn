import torch
from torch.nn import Module
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


class JapaneseVGG16(Module):

    def forward(self, *input):
        pass