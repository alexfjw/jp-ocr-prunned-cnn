from src.datasets import Etl2Dataset, Etl9bDataset
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from torchvision import models, transforms
from utils.model_selection import stratified_test_split
from sklearn.metrics import f1_score
import argparse
import os, sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import tqdm

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

etl9b_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # duplicate the channels
        transforms.ToTensor(),
        transforms.Normalize((Etl9bDataset.mean, Etl9bDataset.mean, Etl9bDataset.mean),
                             (Etl9bDataset.std, Etl9bDataset.std, Etl9bDataset.std))
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3), # duplicate the channels
        transforms.ToTensor(),
        transforms.Normalize((Etl9bDataset.mean, Etl9bDataset.mean, Etl9bDataset.mean),
                             (Etl9bDataset.std, Etl9bDataset.std, Etl9bDataset.std))
    ])
}


def get_etl2_dataloaders():
    """
    returns train & test dataloaders for etl2 dataset
    """
    etl2 = Etl2Dataset(train_transforms=etl2_transforms['train'],
                       test_transforms=etl2_transforms['test'])
    train_indices, val_indices, test_indices, _, _, _ = \
        stratified_test_split(etl2, test_size=0.2, val_size=0.2)

    train_dataloader = DataLoader(etl2,
                                  batch_sampler=BatchSampler(SubsetRandomSampler(train_indices), 32, False),
                                  num_workers=2)
    val_dataloader = DataLoader(etl2,
                                batch_sampler=BatchSampler(SubsetRandomSampler(val_indices), 32, False),
                                num_workers=2)
    test_dataloader = DataLoader(etl2,
                                 batch_sampler=BatchSampler(SubsetRandomSampler(test_indices), 32, False),
                                 num_workers=2)

    return {'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader}, len(etl2.classes)


def train_model(model, dataloaders):
    print('training model:', model)

    # Checks if GPU is available
    use_gpu = torch.cuda.is_available()
    since = time.time()

    # Keeps track of best parameters and f1 score from validation phase
    best_model_wts = model.state_dict()
    best_f1 = 0.0

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    num_epochs=25

    print()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        epoch_start = time.time()

        # Train and validate for each epoch
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train(True)
            else:
                model.train(False)

            dataloader = dataloaders[phase]

            # Keeps track of epoch loss, labels vs predictions
            running_loss = 0.0

            running_labels = torch.LongTensor()
            running_predictions = torch.LongTensor()

            if use_gpu:
                model = model.cuda()

            # Iterate data using dataloaders
            for data in tqdm.tqdm(dataloader):
                inputs, labels = data

                # Wrap inputs and labels in Variables
                if use_gpu:
                    is_val = phase == 'val'
                    inputs = Variable(inputs.cuda(), volatile=is_val)
                    labels = Variable(labels.cuda(), volatile=is_val)
                else:
                    inputs, labels = Variable(inputs), Variable(labels)


                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward and loss calculation
                outputs = model(inputs)
                _, pred_indices = torch.max(outputs.data, 1)
                loss = criterion(outputs, labels)

                # Backward and optimize if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data[0]
                running_labels = torch.cat((running_labels, labels.data.cpu()), 0)
                running_predictions = torch.cat((running_predictions, pred_indices.cpu()), 0)

                del inputs, labels

            loader_size = len(dataloader.batch_sampler.sampler)
            epoch_loss = running_loss / loader_size

            # Calculate f1_score using true labels and predictions
            epoch_f1 = f1_score(running_labels.numpy(), running_predictions.numpy(), average='macro')

            print('{} Loss: {:.4f} F1: {:.4f}'.format(
                phase, epoch_loss, epoch_f1))

            # Update best parameters and f1 score if in validation phase
            if phase == 'val' and epoch_f1 > best_f1:
                best_f1 = epoch_f1
                best_model_wts = model.state_dict()

        epoch_elapsed = time.time() - epoch_start
        print('Epoch {} took in {:.0f}m {:.0f}s'.format(
            epoch, epoch_elapsed // 60, epoch_elapsed % 60))
        print()

    # Compute total time
    time_elasped = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elasped // 60, time_elasped % 60))
    print('Best val F1: {:4f}'.format(best_f1))

    # Return model with optimized parameters
    model.load_state_dict(best_model_wts)
    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--prune', dest='prune', action='store_true')
    parser.add_argument('--model')
    parser.add_argument('--etl2', dest='etl2')
    parser.add_argument('--etl2_9b', dest='etl2_9b')
    parser.add_argument('--train_path', type=str, default='train')
    parser.add_argument('--test_path', type=str, default='prune')
    parser.set_defaults(train=False)
    parser.set_defaults(prune=False)

    return parser.parse_args()


def vgg_model(num_classes):
    model = models.vgg16(True)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model


def custom_model(num_classes):
    # TODO: add custom model
    pass


def main():
    args = get_args()

    data_loaders, num_classes = get_etl2_dataloaders()

    if args.train:
        model = vgg_model(num_classes) if args.model == "vgg16" \
            else custom_model(num_classes)
        model = train_model(model, data_loaders)
        torch.save(model, 'models/vgg16_etl2')

if __name__ == '__main__':
    sys.exit(main())
