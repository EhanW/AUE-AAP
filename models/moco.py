import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50
from .vgg import vgg19
from .mobilenetv2 import MobileNetV2
from .densenet import DenseNet121


class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, backbone: str, projection_dim=128, K=65536, m=0.999, T=0.07, allow_mmt_grad=False, cifar_conv=True):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()
        self.cifar_conv = cifar_conv
        self.allow_mmt_grad = allow_mmt_grad
        self.backbone = backbone
        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = self._get_encoder()
        self.encoder_k = self._get_encoder()

        feature_dim = self.encoder_q.fc.in_features

        self.encoder_q.fc = nn.Sequential(
            nn.Linear(feature_dim, 2048), nn.ReLU(), nn.Linear(2048, projection_dim)
        )
        self.encoder_k.fc = nn.Sequential(
            nn.Linear(feature_dim, 2048), nn.ReLU(), nn.Linear(2048, projection_dim)
        )

        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(projection_dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

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
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        # assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        l = self.queue[:, ptr : ptr + batch_size].shape[1]
        self.queue[:, ptr : ptr + batch_size] = keys.T[:, :l]
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, x):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        im_q, im_k = x[::2], x[1::2]
        # compute query features
        q = self.encoder_q(im_q)  # queries: NxC
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.set_grad_enabled(self.allow_mmt_grad):  # no gradient to keys
            k = self.encoder_k(im_k)  # keys: NxC
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        return logits, labels
