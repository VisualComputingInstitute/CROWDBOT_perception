from torchvision.models.resnet import ResNet, Bottleneck
import torch.nn as nn
from models.utils import weights_init_classifier, weights_init_kaiming
from models import register_model
import builders.merging_block_builder as merging_block_builder


class ClassificationBranch(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, num_classes, bias=False)
        self.linear.apply(weights_init_classifier)
        self.batchnorm = nn.BatchNorm1d(input_dim)
        self.batchnorm.bias.requires_grad_(False) # no shift
        self.batchnorm.apply(weights_init_kaiming)
        self.relu = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool2d(1)

    def forward(self, x, endpoints):
        x = self.pooling(x)
        x = x.view(x.size(0), -1)
        x = self.relu(x)
        x = self.batchnorm(x)
        # emb = x
        x = self.linear(x)
        endpoints['softmax'] = [x]
        return endpoints

    @staticmethod
    def create_endpoints():
        return {'softmax': None}

    @staticmethod
    def build(cfg):
        input_dim = cfg['input_dim']
        num_classes = cfg['num_classes']
        return ClassificationBranch(input_dim, num_classes)


@register_model('classification')
class Classification(ResNet):
    @property
    def dimensions(self):
        return {"emb": (self.emb_dim,)}

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['emb'] = None
        endpoints['softmax'] = None
        return endpoints

    def __init__(self, block, layers, num_classes, merging_block=None):
        """Initializes original ResNet and overwrites fully connected layer."""

        super().__init__(block, layers, 1)  # 0 classes thows an error
        self.fc = None
        # reset inplanes
        self.inplanes = 256 * block.expansion
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1)

        self.num_classes = num_classes
        self.classification = ClassificationBranch(self.inplanes, self.num_classes)
        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.endpoints = self.create_endpoints()
        self.name = "classification"
        self.merging_block = merging_block
        self.emb_dim = 2048

    def forward(self, x, endpoints):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pooling(x)
        x = x.view(x.shape[0], -1)
        endpoints["emb"] = x
        soft, soft_emb = self.classification(x)
        endpoints["softmax"] = [soft]
        # if not self.training:
        #    endpoints["emb"] = self.merging_block(endpoints)
        return endpoints

    @staticmethod
    def build(cfg):
        num_classes = cfg['num_classes']
        merging_block = merging_block_builder.build(cfg['merging_block'])
        model = Classification(Bottleneck, [3, 4, 6, 3], num_classes, merging_block)
        skips = ['fc']
        duplicate = []
        return model, skips, duplicate
