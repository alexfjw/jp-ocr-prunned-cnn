import torch.nn as nn
import itertools
import numpy as np
from utils.iter import grouper


def prune_model(model:nn.Module, dataloaders, prune_ratio=0.5, finetuning_passes=10):
    model.train()
    pruning_iterations = estimate_pruning_iterations(model, prune_ratio)
    # from dataloader, group into 11s & cycle
    data = itertools.cycle(grouper(dataloaders['train'], finetuning_passes+1))

    for i in range(pruning_iterations):
        prune_data, *finetune_data = next(data)
        prune_step(model, prune_data)
        finetune_step(model, finetune_data)

# test all of the below later
def estimate_pruning_iterations(model, prune_ratio):
    num_feature_maps = get_num_feature_maps(model)
    num_params = get_num_parameters(model)
    params_per_map = num_feature_maps // num_params

    return np.ceil(num_params * prune_ratio / params_per_map)


def get_num_feature_maps(model):
    conv2ds = {module for module in model.modules() if issubclass(module, nn.Conv2d)}
    return np.sum({conv2d.out_channels for conv2d in conv2ds})


def get_num_parameters(model):
    # get total number of variables from all conv2d featuremaps
    conv2d_parameters = {module.parameters() for module in model.modules() if issubclass(module, nn.Conv2d)}
    param_objs = itertools.chain(*conv2d_parameters)

    return np.sum({np.prod(np.array(p.size())) for p in param_objs})


def prune_step(model:nn.Module, data):
    result = model(data)
    result.backward()
    # model should have everything populated now
    # for each conv2, gather the featuremaps, sort and so on...
    # throw this into the model instead...


def finetune_step(model, data):
    for x in data:
        # do some finetuning
