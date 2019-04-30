import torch.nn as nn
from models import register_model, BaseModel
from builders import model_builder

def build_pooling_layer(name):
    if name == 'max':
        return nn.AdaptiveMaxPool2d(1)
    elif name == 'avg':
        return nn.AdaptiveAvgPool2d(1)
    elif name == 'combined':
        return CombinedPooling()
    else:
        raise ValueError


class CombinedPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pooling = nn.AdaptiveMaxPool2d(1)
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        max_pooled = self.max_pooling(x)
        avg_pooled = self.avg_pooling(x)

        return max_pooled + avg_pooled


class BaselineReidBranch(nn.Module):
    @staticmethod
    def create_endpoints():
        return {"triplet": None}

    @property
    def dimensions(self):
        return {}

    def __init__(self, pooling):
        super(BaselineReidBranch, self).__init__()
        self.pooling = pooling

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        endpoints['triplet'] = [x]
        emb = x
        return endpoints, emb

    @staticmethod
    def build(cfg):
        pooling = build_pooling_layer(cfg['pooling'])
        return BaselineReidBranch(pooling)


@register_model('baseline')
class Baseline(BaseModel):

    @property
    def dimensions(self):
        return {"emb": (self.dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['triplet'] = None
        endpoints['emb'] = None
        return endpoints

    def __init__(self, backbone, reid_branch):
        """Initializes original ResNet and overwrites fully connected layer."""

        BaseModel.__init__(self) # 2 classes thows an error
        self.dim = 2048
        self.endpoints = self.create_endpoints()
        self.reid_branch = reid_branch
        self.backbone = backbone

    def forward(self, x, endpoints):
        x = self.backbone(x)
        endpoints, x = self.reid_branch(x, endpoints)
        endpoints["emb"] = x
        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        reid_branch = BaselineReidBranch.build(cfg)
        model = Baseline(backbone, reid_branch)
        skips = ["fc"]
        return model, skips, []
