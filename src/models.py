import torch.nn as nn
import src.prunable_nn as pnn
import utils
from src.datasets import *
import torch.utils.model_zoo as model_zoo
from torchvision import models, transforms

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
}

def vgg_model(num_classes):
    model = models.VGG(make_layers(cfg['A'], batch_norm=True))
    model.load_state_dict(model_zoo.load_url(model_urls['vgg11_bn']))

    model.classifier = nn.Sequential(
        pnn.PLinear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model, 'vgg11_bn'

# taken from pytorch's vgg.py
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
}

# taken from pytorch's vgg.py
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = pnn.PConv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, pnn.PBatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def chinese_model(num_classes):
    return ChineseNet(num_classes), 'chinese_net'


class ChineseNet(nn.Module):
    # inspired by https://arxiv.org/abs/1702.07975, used for chinese ocr
    def __init__(self, num_classes):
        super(ChineseNet, self).__init__()
        self.features = self.make_layers()
        self.classifier = nn.Sequential(
            pnn.PLinear(384*2*2, 1024),
            pnn.PBatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(),
            nn.Linear(1024, num_classes)
        )

    def make_layers(self):
        layers = []
        config = [96, 'M', 128, 'M', 160, 'M', 256, 256, 'M', 384, 384, 'M']
        in_channels = 1
        for v in config:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=3, stride=2)]
            else:
                conv2d = pnn.PConv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, pnn.PBatchNorm2d(v), nn.PReLU()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

