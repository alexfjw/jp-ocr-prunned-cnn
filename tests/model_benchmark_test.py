from utils.model_utils import benchmark
from src.nn.models import ChineseNet
from src.data.dataloaders import *
import torch


# load dataloader
dataloaders, _ = get_etl2_dataloaders('chinese_net')

# load model
model = ChineseNet(3156)
model.load_state_dict(torch.load('trained_models/chinese_net_etl2_9g.weights'))

# benchmark
benchmark(model, dataloaders['val'], 'benchmark_test')

