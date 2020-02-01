import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet18


def model_fn(model, data):
    tb_dict, disp_dict = {}, {}

    polar_grid = data['polar_grid']
    polar_grid = torch.from_numpy(polar_grid).cuda(non_blocking=True).float()

    # Forward pass
    pred_cls, pred_reg = model(polar_grid)

    target_cls, target_reg = data['target_cls'], data['target_reg']
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).long()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

    n_batch, n_cutout = target_cls.shape[:2]

    # cls loss
    target_cls = target_cls.view(n_batch*n_cutout)
    pred_cls = pred_cls.view(n_batch*n_cutout, -1)
    cls_loss = F.cross_entropy(pred_cls, target_cls, reduction='mean')
    total_loss = cls_loss

    # number fg points
    fg_mask = target_cls.ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (n_batch * n_cutout)

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(n_batch*n_cutout, -1)
        pred_reg = pred_reg.view(n_batch*n_cutout, -1)
        reg_loss = F.mse_loss(pred_reg[fg_mask], target_reg[fg_mask],
                              reduction='none')
        reg_loss = torch.sqrt(torch.sum(reg_loss, dim=1)).mean()
        total_loss = reg_loss + cls_loss
    else:
        reg_loss = 0

    disp_dict['loss'] = total_loss

    tb_dict['cls_loss'] = cls_loss
    tb_dict['reg_loss'] = reg_loss
    tb_dict['fg_ratio'] = fg_ratio

    return total_loss, tb_dict, disp_dict


def model_fn_eval(model, data):
    total_loss, tb_dict, disp_dict = model_fn(model, data)
    if tb_dict['fg_ratio'] == 0.0:
        del tb_dict['reg_loss']  # So that it's not summed in caucluating epoch average

    return total_loss, tb_dict, disp_dict


class PolarDROW(nn.Module):
    def __init__(self, num_scans, input_size=(300, 450), pretrained=False, *args, **kwargs):
        super(PolarDROW, self).__init__(*args, **kwargs)
        self.num_scans = num_scans

        net = resnet18(pretrained=pretrained, progress=True)  # 512 channel, 32x downsample
        net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(net.conv1.weight)
        del net.avgpool
        del net.fc

        self.net = net

        self.head1 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        fc_in_channel = int(
                512 * math.ceil(input_size[0]/64) * math.ceil(input_size[1]/64))
        self.head2 = nn.Sequential(
                nn.Linear(fc_in_channel, 4096),
                nn.Linear(4096, input_size[1]*6))
        nn.init.kaiming_normal_(self.head1[0].weight)
        nn.init.kaiming_normal_(self.head2[0].weight)
        nn.init.kaiming_normal_(self.head2[1].weight)

    def forward(self, x):
        n_batch, _, n_range, n_pts = x.shape

        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        x = self.net.layer4(x)

        x = self.head1(x)
        x = x.view(n_batch, -1)
        x = self.head2(x)
        x = x.view(n_batch, n_pts, 6)

        pred_cls, pred_reg = x[..., :4], x[..., 4:]

        return pred_cls, pred_reg