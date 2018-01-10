from utils.model_utils import benchmark
from src.nn.models import ChineseNet
from src.data.dataloaders import get_etl2_9g_dataloaders
import torch


# load dataloader
dataloaders, num_classes = get_etl2_9g_dataloaders('chinese_net')

# load model
model = ChineseNet(num_classes)
model.load_state_dict(torch.load('trained_models/chinese_net_etl2_9g.weights'))

# benchmark
benchmark(model, dataloaders['val'], 'benchmark_test')

