import torch.nn as nn
import utils
from src.datasets import *
from torchvision import models, transforms


def vgg_model(num_classes):
    model = models.vgg11_bn(True)
    model.classifier = nn.Sequential(
        nn.Linear(512 * 7 * 7, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(4096, num_classes),
    )
    return model, 'vgg11_bn'


def chinese_model(num_classes):
        return ChineseNet(num_classes), 'chinese_net'


class ChineseNet(nn.Module):
    # inspired by https://arxiv.org/abs/1702.07975, used for chinese ocr
    def __init__(self, num_classes):
        super(ChineseNet, self).__init__()
        self.features = self.make_layers()
        self.classifier = nn.Sequential(
            nn.Linear(384*2*2, 1024),
            nn.BatchNorm1d(1024),
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
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.BatchNorm2d(v), nn.PReLU()]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



