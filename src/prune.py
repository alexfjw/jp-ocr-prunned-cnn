import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from tqdm import tqdm
from utils.model_utils import benchmark
from utils.iter import grouper
from torch.autograd import Variable


def prune_model(model:nn.Module, dataloaders, prune_ratio=0.5, finetuning_passes=10):
    use_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()
    pruning_iterations = estimate_pruning_iterations(model, prune_ratio)
    checkpoint = pruning_iterations // 10

    # from dataloader, group into 11s & cycle
    dataloaders['train'].dataset.train = True
    data = itertools.cycle(grouper(dataloaders['train'], finetuning_passes+1))

    # benchmark(model, dataloaders['val'], 'before pruning')

    for i in tqdm(range(pruning_iterations)):
        if use_gpu:
            model = model.cuda()
        model.train()
        prune_data, *finetune_data = next(data)
        prune_step(model, prune_data, criterion, use_gpu)
        finetune_step(model, finetune_data, criterion, use_gpu)

        # check progress every 10% of the journey
        if (i % checkpoint) == (checkpoint - 1):
            benchmark(model, dataloaders['val'], f'pruning, {i}/{pruning_iterations} iterations')


# test all of the below later
def estimate_pruning_iterations(model, prune_ratio):
    num_feature_maps = get_num_prunable_feature_maps(model)
    num_params = get_num_parameters(model)
    params_per_map = num_params // num_feature_maps

    return int(np.ceil(num_params * prune_ratio / params_per_map))


def get_num_prunable_feature_maps(model):
    conv2ds = {module for module in model.modules() if issubclass(type(module), nn.Conv2d)}
    return np.sum(conv2d.out_channels for conv2d in conv2ds)


def get_num_parameters(model):
    # get total number of variables from all conv2d featuremaps
    conv2d_parameters = (module.parameters() for module in model.modules() if issubclass(type(module), nn.Conv2d))
    param_objs = itertools.chain(*conv2d_parameters)

    return np.sum(np.prod(np.array(p.size())) for p in param_objs)


def prune_step(model:nn.Module, data, criterion, use_gpu):
    inputs, labels = data

    # Wrap inputs and labels in Variables
    if use_gpu:
        inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
    else:
        inputs, labels = Variable(inputs), Variable(labels)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    model.prune()

    del inputs, labels


def finetune_step(model, data, criterion, use_gpu):
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    for x in data:
        inputs, labels = x
        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        del inputs, labels
