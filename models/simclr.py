import torch.nn as nn
from torchvision.models import resnet18, resnet50
from .vgg import vgg19
from .mobilenetv2 import MobileNetV2
from .densenet import DenseNet121


class SimCLR(nn.Module):
    def __init__(self, backbone: str, projection_dim=128, cifar_conv=True):
        super().__init__()
        self.cifar_conv = cifar_conv
        self.backbone = backbone
        self.enc = self._get_encoder()
        self.feature_dim = self.enc.fc.in_features
        self.enc.fc = nn.Sequential(nn.Linear(self.feature_dim, 2048),
                                    nn.BatchNorm1d(2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, projection_dim),
                                    nn.BatchNorm1d(projection_dim)
                                    )

    def _get_encoder(self):
        if self.backbone == 'resnet18':
            encoder = resnet18()
            if self.cifar_conv:
                encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                encoder.maxpool = nn.Identity()
        elif self.backbone == 'resnet50':
            encoder = resnet50()
            if self.cifar_conv:
                encoder.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
                encoder.maxpool = nn.Identity()
        elif self.backbone == 'vgg19':
            encoder = vgg19()
        elif self.backbone == 'mobilenet':
            encoder = MobileNetV2()
        elif self.backbone == 'densenet121':
            encoder = DenseNet121()
        else:
            raise AssertionError('model is not defined')
        return encoder

    def forward(self, x):
        projection = self.enc(x)
        return projection

