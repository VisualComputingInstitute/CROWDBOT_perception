import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
# From https://github.com/mbsariyildiz/focal-loss.pytorch/blob/master/focalloss.py
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = logpt.exp()

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * at

        loss = -1 * (1 - pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


def model_fn(model, data):
    tb_dict, disp_dict = {}, {}

    net_input = data['input']
    net_input = torch.from_numpy(net_input).cuda(non_blocking=True).float()

    # Forward pass
    pred_cls, pred_reg = model(net_input)

    target_cls, target_reg = data['target_cls'], data['target_reg']
    target_cls = torch.from_numpy(target_cls).cuda(non_blocking=True).long()
    target_reg = torch.from_numpy(target_reg).cuda(non_blocking=True).float()

    n_batch, n_pts = target_cls.shape[:2]

    # cls loss
    target_cls = target_cls.view(n_batch * n_pts)
    pred_cls = pred_cls.view(n_batch * n_pts, -1)
    if model.cls_loss is not None:
        cls_loss = model.cls_loss(pred_cls, target_cls)
    else:
        cls_loss = F.cross_entropy(pred_cls, target_cls, reduction='mean')
    total_loss = cls_loss

    # number fg points
    fg_mask = target_cls.ne(0)
    fg_ratio = torch.sum(fg_mask).item() / (n_batch * n_pts)

    # reg loss
    if fg_ratio > 0.0:
        target_reg = target_reg.view(n_batch * n_pts, -1)
        pred_reg = pred_reg.view(n_batch * n_pts, -1)
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
