from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F


def model_fn(model, data):
    tb_dict, disp_dict = {}, {}

    cutout_input = data['cutout']
    cutout_input = torch.from_numpy(cutout_input).cuda(non_blocking=True).float()

    # Forward pass
    pred_cls, pred_reg = model(cutout_input)

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


def lbt_init(mod, init_=None, bias=0):
    # Copy paste from lbtoolbox.pytorch.lbt
    """ Initializes `mod` with `init` and returns it.
    Also sets the bias to the given constant value, if available.

    Useful for the `Sequential` constructor and friends.
    """
    if init_ is not None and getattr(mod, 'weight', None) is not None:
        init_(mod.weight)
    if getattr(mod, 'bias', None) is not None:
        torch.nn.init.constant_(mod.bias, bias)
    return mod


class FastDROWNet3LF2p(nn.Module):
    def __init__(self, num_scans, dropout=0.5, sequential_inference=False, *args, **kwargs):
        super(FastDROWNet3LF2p, self).__init__(*args, **kwargs)
        self._sequential_inference = sequential_inference
        self.num_scans = num_scans

        # In case of sequential input, save previous intermediate features to
        # imporve inference speed (not used in training)
        if self._sequential_inference:
            self._hx_deque = deque(maxlen=num_scans)

        self.dropout = dropout
        self.conv1a = nn.Conv1d(1, 64, kernel_size=3, padding=1)
        self.bn1a = nn.BatchNorm1d(64)
        self.conv1b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm1d(64)
        self.conv1c = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn1c = nn.BatchNorm1d(128)
        self.conv2a = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2a = nn.BatchNorm1d(128)
        self.conv2b = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(128)
        self.conv2c = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2c = nn.BatchNorm1d(256)
        self.conv3a = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3a = nn.BatchNorm1d(256)
        self.conv3b = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(256)
        self.conv3c = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn3c = nn.BatchNorm1d(512)
        self.conv4a = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.bn4a = nn.BatchNorm1d(256)
        self.conv4b = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.bn4b = nn.BatchNorm1d(128)
        self.conv4p = nn.Conv1d(128, 4, kernel_size=1)  # probs
        self.conv4v = nn.Conv1d(128, 2, kernel_size=1)  # vote

        self.reset_parameters()

    def forward(self, x):
        n_batch, n_cutout, n_scan, n_points = x.shape

        def trunk_forward(x):
            x = F.leaky_relu(self.bn1a(self.conv1a(x)), 0.1)
            x = F.leaky_relu(self.bn1b(self.conv1b(x)), 0.1)
            x = F.leaky_relu(self.bn1c(self.conv1c(x)), 0.1)
            x = F.max_pool1d(x, 2)  # 24
            x = F.dropout(x, p=self.dropout, training=self.training)

            x = F.leaky_relu(self.bn2a(self.conv2a(x)), 0.1)
            x = F.leaky_relu(self.bn2b(self.conv2b(x)), 0.1)
            x = F.leaky_relu(self.bn2c(self.conv2c(x)), 0.1)
            x = F.max_pool1d(x, 2)  # 12
            x = F.dropout(x, p=self.dropout, training=self.training)
            return x

        if not self.training and self._sequential_inference:
            assert n_batch == n_scan == 1
            x = torch.squeeze(x, dim=0)
            hx = trunk_forward(x)
            self._hx_deque.append(hx)
            while len(self._hx_deque) < self._hx_deque.maxlen:
                self._hx_deque.append(hx)
            x = torch.stack(list(self._hx_deque), dim=0).sum(dim=0)
            x = torch.unsqueeze(x, dim=0)
        else:
            x = x.view(n_batch*n_cutout*n_scan, 1, n_points)
            x = trunk_forward(x)
            x = x.view(n_batch, n_cutout, n_scan, -1, x.shape[-1])
            x = torch.sum(x, dim=2)  # batch, cutout, feature, points

        x = x.view(n_batch*n_cutout, -1, x.shape[-1])

        x = F.leaky_relu(self.bn3a(self.conv3a(x)), 0.1)
        x = F.leaky_relu(self.bn3b(self.conv3b(x)), 0.1)
        x = F.leaky_relu(self.bn3c(self.conv3c(x)), 0.1)
        x = F.max_pool1d(x, 2)  # 6
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.leaky_relu(self.bn4a(self.conv4a(x)), 0.1)
        x = F.leaky_relu(self.bn4b(self.conv4b(x)), 0.1)
        x = F.avg_pool1d(x, 6)  # Due to the arch, output has spatial size 1
        logits = self.conv4p(x).view(n_batch, n_cutout, 4)
        votes = self.conv4v(x).view(n_batch, n_cutout, 2)

        return logits, votes

    def reset_parameters(self):
        lbt_init(self.conv1a, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv1b, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv1c, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv2a, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv2b, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv2c, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv3a, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv3b, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv3c, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv4a, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv4b, lambda t: nn.init.kaiming_normal_(t, a=0.1), 0)
        lbt_init(self.conv4p, lambda t: nn.init.constant_(t, 0), 0)
        lbt_init(self.conv4v, lambda t: nn.init.constant_(t, 0), 0)
        nn.init.constant_(self.bn1a.weight, 1)
        nn.init.constant_(self.bn1b.weight, 1)
        nn.init.constant_(self.bn1c.weight, 1)
        nn.init.constant_(self.bn2a.weight, 1)
        nn.init.constant_(self.bn2b.weight, 1)
        nn.init.constant_(self.bn2c.weight, 1)
        nn.init.constant_(self.bn3a.weight, 1)
        nn.init.constant_(self.bn3b.weight, 1)
        nn.init.constant_(self.bn3c.weight, 1)
        nn.init.constant_(self.bn4a.weight, 1)
        nn.init.constant_(self.bn4b.weight, 1)
