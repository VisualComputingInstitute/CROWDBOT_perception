import torch.nn as nn
import torch.nn.functional as F
import models.panoptic.resnet_fpn as FPN
from models import BaseModel
from models import register_model


class UpsampleBlock(nn.Module):

    def __init__(self, input_dim, output_dim, upsample=True):
        super(UpsampleBlock, self).__init__()

        num_groups = 16
        self.upsample = upsample
        self.conv1   = nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=1, padding=1)
        self.gn1     = nn.GroupNorm(num_groups, output_dim)

    def forward(self, x):
        f = F.relu(self.gn1(self.conv1(x)))

        if self.upsample:
            out = F.upsample(f, scale_factor=2, mode='bilinear')
        else:
            out = f

        return out


@register_model("resnet-fpn-sem-head")
class SemNet(BaseModel):

    def __init__(self, num_classes, resnet=50):
        super(SemNet, self).__init__()

        self.fpn = self._choose_resnet(resnet)
        self.endpoints = self.create_endpoints()

        self.num_classes = num_classes

        input_dim = 256
        output_dim = 128

        self.p5upsample = self._make_block(3, input_dim, output_dim)
        self.p4upsample = self._make_block(2, input_dim, output_dim)
        self.p3upsample = self._make_block(1, input_dim, output_dim)
        self.p2upsample = nn.Sequential(UpsampleBlock(input_dim, output_dim, upsample=False))

        self.conv1 = nn.Conv2d(128, self.num_classes, kernel_size=1, stride=1)

    def _choose_resnet(self, number):
        if number == 50:
            return FPN.FPN50(pretrained=True)
        elif number == 101:
            return FPN.FPN101(pretrained=True)
        elif number == 152:
            return FPN.FPN152(pretrained=True)
        else:
            raise ValueError("Requested ResNet is not available")


    def _make_block(self, num_upsample, input_dim, output_dim):
        layers = []

        for i in range(num_upsample):
            if i < num_upsample - 1:
                layers.append(UpsampleBlock(input_dim, input_dim, upsample=True))
            else:
                layers.append(UpsampleBlock(input_dim, output_dim, upsample=False))

        return nn.Sequential(*layers)


    def upsample_to_fixed_size(self, x, y):
        """Necessary to get to same size"""
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear')

    def forward(self, data, endpoints):
        device = next(self.parameters()).device
        x = data['img'].to(device, non_blocking=True)

        (p2, p3, p4, p5) = self.fpn(x)
        f5 = self.p5upsample(p5)
        f5 = self.upsample_to_fixed_size(f5, p2)
        f4 = self.p4upsample(p4)
        f4 = self.upsample_to_fixed_size(f4, p2)
        f3 = self.p3upsample(p3)
        f3 = self.upsample_to_fixed_size(f3, p2)
        f2 = self.p2upsample(p2)

        f1 = f2 + f3 + f4 + f5
        f1 = self.conv1(f1)

        # orig_size
        _, _, h, w = x.size()
        logits = F.upsample(f1, size=(h, w), mode='bilinear')
        endpoints['sem-logits'] = logits

        return endpoints

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['sem-logits'] = None
        return endpoints

    @staticmethod
    def build(cfg):
        model = SemNet(cfg['num_classes'], cfg['resnet'])
        skips = ["fc"]
        duplicates = []
        return model, skips, duplicates

    def train(self, mode=True):
        super().train(mode)
        self.freeze_batchnorm()
        return self

    def freeze_batchnorm(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def unfreeze_batchnorm(self):
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.train()
                for param in layer.parameters():
                    param.requires_grad = True

