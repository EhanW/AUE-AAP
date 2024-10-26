import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from .vgg import vgg19
from .mobilenetv2 import MobileNetV2
from .densenet import DenseNet121


class BYOL(nn.Module):
    def __init__(self, backbone: str, projection_dim=128, m=0.99, allow_mmt_grad=False, cifar_conv=True):
        super(BYOL, self).__init__()
        self.cifar_conv = cifar_conv
        self.allow_mmt_grad = allow_mmt_grad
        self.m = m
        self.backbone = backbone
        self.online_encoder = self._get_encoder()
        self.target_encoder = self._get_encoder()
        feature_dim = self.online_encoder.fc.in_features

        self.online_encoder.fc = nn.Sequential(
            nn.Linear(feature_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, projection_dim)
        )
        self.target_encoder.fc = nn.Sequential(
            nn.Linear(feature_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, projection_dim)
        )

        self.online_predictor = nn.Sequential(
            nn.Linear(projection_dim, 2048), nn.BatchNorm1d(2048), nn.ReLU(), nn.Linear(2048, projection_dim)
        )
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data.copy_(online_param.data)
            target_param.requires_grad = False

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

    @torch.no_grad()
    def momentum_update_target_encoder(self):
        for online_param, target_param in zip(
            self.online_encoder.parameters(), self.target_encoder.parameters()
        ):
            target_param.data = target_param.data * self.m + online_param.data * (1.0 - self.m)

    def forward(self, im):
        online_output = self.online_encoder(im)
        online_output = self.online_predictor(online_output)

        with torch.set_grad_enabled(self.allow_mmt_grad):
            target_output = self.target_encoder(im)
        return online_output, target_output
