from collections import deque
from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss_utils import FocalLoss


def _conv(in_channel, out_channel, kernel_size, padding):
    return nn.Sequential(nn.Conv1d(in_channel, out_channel,
                                   kernel_size=kernel_size, padding=padding),
                         nn.BatchNorm1d(out_channel),
                         nn.LeakyReLU(negative_slope=0.1, inplace=True))


def _conv3x3(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=3, padding=1)


def _conv1x1(in_channel, out_channel):
    return _conv(in_channel, out_channel, kernel_size=1, padding=1)


class _TemporalGate(nn.Module):
    def __init__(self, n_scans, n_pts, n_channel):
        super(_TemporalGate, self).__init__()
        self.conv1 = nn.Conv1d(n_channel, 128, kernel_size=n_pts, padding=0)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=n_scans, padding=0)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc = nn.Linear(64, n_scans)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)


    def forward(self, x):
        n_batch, n_scans, n_channel, n_pts = x.shape

        out = x.view(n_batch * n_scans, n_channel, n_pts)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = out.view(n_batch, n_scans, 128).permute(0, 2, 1)  # (batch, feature, scans)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out).view(n_batch, 64)  # (batch, feature)

        out = self.fc(out)
        out = F.softmax(out, dim=1)  # (batch, scans)

        return out


class DROW(nn.Module):
    def __init__(self, dropout=0.5, num_scans=5, num_pts=48, temporal_gating=True,
                 sequential_inference=False, focal_loss_gamma=0.0):
        super(DROW, self).__init__()
        self._sequential_inference = sequential_inference

        # In case of sequential input, save previous intermediate features to
        # imporve inference speed (not used in training)
        if self._sequential_inference:
            self._hx_deque = deque(maxlen=num_scans)

        self.dropout = dropout

        self.conv_block_1 = nn.Sequential(_conv3x3(1, 64),
                                          _conv3x3(64, 64),
                                          _conv3x3(64, 128))
        self.conv_block_2 = nn.Sequential(_conv3x3(128, 128),
                                          _conv3x3(128, 128),
                                          _conv3x3(128, 256))
        self.conv_block_3 = nn.Sequential(_conv3x3(256, 256),
                                          _conv3x3(256, 256),
                                          _conv3x3(256, 512))
        self.conv_block_4 = nn.Sequential(_conv3x3(512, 256),
                                          _conv3x3(256, 128))

        if temporal_gating:
            self.gate = _TemporalGate(num_scans, int(ceil(num_pts / 4)), 256)
        else:
            self.gate = None

        self.conv_cls = nn.Conv1d(128, 4, kernel_size=1)  # probs
        self.conv_reg = nn.Conv1d(128, 2, kernel_size=1)  # vote

        if focal_loss_gamma > 0.0:
            self.cls_loss = FocalLoss(gamma=focal_loss_gamma)
        else:
            self.cls_loss = None

        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, a=0.1, nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _forward_conv(self, x, conv_block):
        out = conv_block(x)
        out = F.max_pool1d(out, kernel_size=2)
        if self.dropout > 0:
            out = F.dropout(out, p=self.dropout, training=self.training)

        return out

    def forward(self, x):
        n_batch, n_cutout, n_scan, n_pts = x.shape

        out = x.view(n_batch * n_cutout * n_scan, 1, n_pts)

        # feature for each cutout
        out = self._forward_conv(out, self.conv_block_1)  # 24
        out = self._forward_conv(out, self.conv_block_2)  # 12

        out = out.view(n_batch * n_cutout, n_scan, *out.shape[-2:])

        if self.gate is not None:
            # temporal gating
            gate = self.gate(out)
            out = out * gate.view(gate.shape[0], gate.shape[1], 1, 1)
            out = torch.sum(out, dim=1)    # (batch*cutout, channel, pts)
        else:
            out = torch.sum(out, dim=1)

        # feature for fused cutout
        out = self._forward_conv(out, self.conv_block_3)  # 6
        out = self.conv_block_4(out)
        out = F.avg_pool1d(out, kernel_size=out.shape[-1])  # (batch*cutout, channel, 1)

        pred_cls = self.conv_cls(out).view(n_batch, n_cutout, 4)
        pred_reg = self.conv_reg(out).view(n_batch, n_cutout, 2)

        return pred_cls, pred_reg