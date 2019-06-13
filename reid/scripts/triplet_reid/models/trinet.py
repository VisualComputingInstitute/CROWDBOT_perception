import torch.nn as nn
from torchvision.models.resnet import ResNet, Bottleneck
from models import register_model

@register_model('trinet')
class TriNet(ResNet):
    """TriNet implementation.

    Replaces the last layer of ResNet50 with two fully connected layers.

    First: 1024 units with batch normalization and ReLU
    Second: 128 units, final embedding.
    """

    def __init__(self, block, layers, dim=128):
        """Initializes original ResNet and overwrites fully connected layer."""

        super(TriNet, self).__init__(block, layers, 1) # 0 classes thows an error
        batch_norm = nn.BatchNorm1d(1024)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, 1024),
            batch_norm,
            nn.ReLU(),
            nn.Linear(1024, dim)
        )
        batch_norm.weight.data.fill_(1)
        batch_norm.bias.data.zero_()
        self.dim = dim
        self.endpoints = self.create_endpoints()


    def forward(self, x, endpoints):
        x = super().forward(x)
        endpoints["emb"] = x
        endpoints["triplet"] = [x]
        return endpoints

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints["emb"] = None
        endpoints["triplet"] = None
        return endpoints

    @property
    def dimensions(self):
        return {"emb": (self.dim,)}


    @staticmethod
    def build(cfg):
        dim = cfg['dim']
        model = TriNet(Bottleneck, [3, 4, 6, 3], dim)
        skips = ['fc']
        return model, skips, []
