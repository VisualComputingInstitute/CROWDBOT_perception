import torch
import torch.nn as nn
import torch.nn.functional as F

from model.loss_utils import FocalLoss


class _ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=None):
        super(_ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, stride=stride,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.dropout = dropout

        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, stride=stride,
                              kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels))


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity

        out = self.relu(out)

        if self.dropout is not None:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out


class PolarDROW(nn.Module):
    def __init__(self, in_channel, dropout=0.5, focal_loss_gamma=0.0):
        super(PolarDROW, self).__init__()
        self.dropout = dropout

        self.conv0 = _ResBlock(in_channel, 64, dropout=dropout)
        self.conv1 = _ResBlock(64, 128, stride=(1, 2), dropout=dropout)
        self.conv2 = _ResBlock(128, 256, stride=(1, 2), dropout=dropout)
        self.conv3 = _ResBlock(256, 512, stride=(1, 2), dropout=dropout)
        self.conv4 = _ResBlock(512, 512, stride=(1, 2), dropout=dropout)

        self.conv_up_3 = _ResBlock(512 + 512, 512, dropout=dropout)
        self.conv_up_2 = _ResBlock(512 + 256, 512, dropout=dropout)
        self.conv_up_1 = _ResBlock(512 + 128, 512, dropout=dropout)
        self.conv_up_0 = _ResBlock(512 + 64, 512)

        self.conv_cls = nn.Conv1d(512, 4, kernel_size=1)
        self.conv_reg = nn.Conv1d(512, 2, kernel_size=1)

        if focal_loss_gamma > 0.0:
            self.cls_loss = FocalLoss(gamma=focal_loss_gamma)
        else:
            self.cls_loss = None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.01, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _upsample(self, x, size):
        return F.interpolate(x, size=size, mode='nearest')


    def forward(self, x):
        x = x.squeeze(dim=1)  # This is the temporal dimension (to be implemented)

        down0 = self.conv0(x)
        down1 = self.conv1(down0)
        down2 = self.conv2(down1)
        down3 = self.conv3(down2)
        down4 = self.conv4(down3)

        up3 = self._upsample(down4, size=down3.shape[-2:])
        up3 = torch.cat((down3, up3), dim=1)
        up3 = self.conv_up_3(up3)

        up2 = self._upsample(up3, size=down2.shape[-2:])
        up2 = torch.cat((down2, up2), dim=1)
        up2 = self.conv_up_2(up2)

        up1 = self._upsample(up2, size=down1.shape[-2:])
        up1 = torch.cat((down1, up1), dim=1)
        up1 = self.conv_up_1(up1)

        up0 = self._upsample(up1, size=down0.shape[-2:])
        up0 = torch.cat((down0, up0), dim=1)
        up0 = self.conv_up_0(up0)

        out = F.max_pool2d(up0, kernel_size=(x.shape[-2], 1))
        out = out.squeeze(dim=2)

        pred_cls = self.conv_cls(out).permute(0, 2, 1).contiguous()
        pred_reg = self.conv_reg(out).permute(0, 2, 1).contiguous()

        return pred_cls, pred_reg