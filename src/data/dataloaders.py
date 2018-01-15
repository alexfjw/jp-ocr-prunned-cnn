from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler, BatchSampler
from utils.model_selection import stratified_test_split
from src.data.datasets import *
from utils.transforms import ToFloat
from torchvision import transforms


def get_etl2_9g_dataloaders(model_type):
    """
    returns train & test dataloaders for etl dataset
    """
    etl2_transforms = transfer_learn_etl2_transforms if model_type == 'vgg' else chinese_transforms_etl2
    etl9_transforms = transfer_learn_etl9g_transforms if model_type == 'vgg' else chinese_transforms_etl9g

    etl2_9g = Etl_2_9G_Dataset(etl2_transforms, etl9_transforms)
    train_indices, val_indices, test_indices, _, _, _ = \
        stratified_test_split(etl2_9g, test_size=0.2, val_size=0.2)

    train_dataloader = DataLoader(etl2_9g,
                                  batch_sampler=BatchSampler(SubsetRandomSampler(train_indices), 32, False),
                                  num_workers=2)
    val_dataloader = DataLoader(etl2_9g,
                                batch_sampler=BatchSampler(SubsetRandomSampler(val_indices), 32, False),
                                num_workers=2)
    test_dataloader = DataLoader(etl2_9g,
                                 batch_sampler=BatchSampler(SubsetRandomSampler(test_indices), 32, False),
                                 num_workers=2)

    return {'train': train_dataloader,
            'val': val_dataloader,
            'test': test_dataloader}, len(etl2_9g.label_encoder.classes_)


def get_etl2_dataloaders(model_type):
    """
    returns train & test dataloaders for etl2 dataset
    """
    transform_group = transfer_learn_etl2_transforms if model_type == 'vgg11_bn' else chinese_transforms_etl2

    etl2 = Etl2Dataset(transform_group['train'], transform_group['test'])

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
            'test': test_dataloader}, len(etl2.label_encoder.classes_)


# ToFloat is required for normalization, tensor may be of type int depending on pil mode
transfer_learn_etl2_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # duplicate the channels
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize((Etl2Dataset.mean, Etl2Dataset.mean, Etl2Dataset.mean),
                             (Etl2Dataset.std, Etl2Dataset.std, Etl2Dataset.std))
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # duplicate the channels
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize((Etl2Dataset.mean, Etl2Dataset.mean, Etl2Dataset.mean),
                             (Etl2Dataset.std, Etl2Dataset.std, Etl2Dataset.std))
    ])
}

transfer_learn_etl9g_transforms = {
    'train': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # duplicate the channels
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize((Etl9GDataset.mean, Etl9GDataset.mean, Etl9GDataset.mean),
                             (Etl9GDataset.std, Etl9GDataset.std, Etl9GDataset.std))
    ]),
    'test': transforms.Compose([
        transforms.Resize(224),
        transforms.Grayscale(num_output_channels=3),  # duplicate the channels
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize((Etl9GDataset.mean, Etl9GDataset.mean, Etl9GDataset.mean),
                             (Etl9GDataset.std, Etl9GDataset.std, Etl9GDataset.std))
    ])
}

chinese_transforms_etl2 = {
    'train': transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize([Etl2Dataset.mean], [Etl2Dataset.std])
    ]),
    'test': transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize([Etl2Dataset.mean], [Etl2Dataset.std])
    ])
}

chinese_transforms_etl9g = {
    'train': transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize([Etl9GDataset.mean], [Etl9GDataset.std])
    ]),
    'test': transforms.Compose([
        transforms.Resize(96),
        transforms.ToTensor(),
        ToFloat,
        transforms.Normalize([Etl9GDataset.mean], [Etl9GDataset.std])
    ])
}