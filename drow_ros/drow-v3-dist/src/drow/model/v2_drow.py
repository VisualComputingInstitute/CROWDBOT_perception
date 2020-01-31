import torch
import torch.nn as nn
import torch.nn.functional as F


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


def apply_dim(fn, dim=0, *inputs):
    return torch.stack([fn(*(a[0] for a in args)) for args in zip(*(inp.split(1, dim=dim) for inp in inputs))], dim=dim)


def apply_dim_keepdim(fn, dim=0, *inputs):
    return torch.cat([fn(*(a for a in args)) for args in zip(*(inp.split(1, dim=dim) for inp in inputs))], dim=dim)


def apply_sum(fn, dim=0, *inputs):
    outs = [fn(*(a for a in args)) for args in zip(*(inp.split(1, dim=dim) for inp in inputs))]
    out = outs[0]
    for o in outs[1:]:
        out = out + o
    return out


def apply_sum2(fn, x):
    (B, T), R = x.shape[:2], x.shape[2:]
    x2 = x.view(B * T, 1, *R)
    x2 = fn(x2)
    x = x2.view(B, T, *x2.shape[1:])
    return torch.sum(x, dim=1)


class DROWNet3LF2p(nn.Module):
    def __init__(self, dropout=0.5, *args, **kwargs):
        super(DROWNet3LF2p, self).__init__(*args, **kwargs)
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

        # x = apply_sum(trunk_forward, x, dim=1)
        x = apply_sum2(trunk_forward, x)

        x = F.leaky_relu(self.bn3a(self.conv3a(x)), 0.1)
        x = F.leaky_relu(self.bn3b(self.conv3b(x)), 0.1)
        x = F.leaky_relu(self.bn3c(self.conv3c(x)), 0.1)
        x = F.max_pool1d(x, 2)  # 6
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.leaky_relu(self.bn4a(self.conv4a(x)), 0.1)
        x = F.leaky_relu(self.bn4b(self.conv4b(x)), 0.1)
        x = F.avg_pool1d(x, 6)
        logits = self.conv4p(x)
        votes = self.conv4v(x)
        return logits[:, :, 0], votes[:, :, 0]  # Due to the arch, output has spatial size 1, so we [0] it.

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