import torch.nn as nn
from torchvision.models import resnet18, resnet50
from .vgg import vgg19
from .mobilenetv2 import MobileNetV2
from .densenet import DenseNet121
import torch.nn.functional as F 

def negcos(p1, p2, z1, z2):
    p1 = F.normalize(p1, dim=1); p2 = F.normalize(p2, dim=1)
    z1 = F.normalize(z1, dim=1); z2 = F.normalize(z2, dim=1)

    return - 0.5 * ((p1*z2.detach()).sum(dim=1).mean() + (p2*z1.detach()).sum(dim=1).mean())

class SimSiam(nn.Module):
    def __init__(self, backbone: str, cifar_conv=True):
        super().__init__()
        self.cifar_conv = cifar_conv
        self.backbone = backbone
        self.enc = self._get_encoder()
        self.feature_dim = self.enc.fc.in_features
        self.criterion = negcos

        self.enc.fc = nn.Sequential(
                    nn.Linear(self.feature_dim, 2048),
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),

                    nn.Linear(2048, 2048),
                    nn.BatchNorm1d(2048),
                )
        
        self.pred_head = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Linear(512, 2048),
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
    
    def forward(self, x1, x2):
        z1 = self.enc(x1)
        z2 = self.enc(x2)
        
        p1 = self.pred_head(z1)
        p2 = self.pred_head(z2)
        
        return self.criterion(p1, p2, z1, z2)

