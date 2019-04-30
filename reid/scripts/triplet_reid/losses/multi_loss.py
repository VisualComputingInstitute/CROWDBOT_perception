import torch
import torch.nn as nn
import logger


class MultiLoss(nn.Module):
    def __init__(self, weight_module, data_controller):
        super().__init__()
        self.weight_module = weight_module
        self.data_controller = data_controller

    def forward(self, endpoints, data):
        if 'split_info' in data:
            # Multi Dataset
            # split info needs to be removed, otherwise it is tried to
            # be collated
            split_info = data.pop('split_info')
            filtered_endpoints = self.data_controller.split(endpoints, split_info)
            filtered_data = self.data_controller.split(data, split_info)
            return self.weight_module(filtered_endpoints, filtered_data, filtered=True)
        else:
            # single Dataset
            return self.weight_module(endpoints, data, filtered=False)


class SingleLoss(nn.Module):
    def __init__(self, loss):
        super().__init__()
        self.loss = loss

    def forward(self, endpoints, data):
        return self.loss(endpoints, data)


class WeightModule(nn.Module):
    def __init__(self, losses):
        super().__init__()
        self.losses = torch.nn.ModuleDict(losses)
        self.logger = logger.get_tensorboard_logger()

    def forward(self, split_endpoints, split_data, filtered):
        overall_loss = 0.0
        for loss_name, loss in self.losses.items():
            if filtered:
                endpoints = split_endpoints[loss_name]
                data = split_data[loss_name]
            else:
                endpoints = split_endpoints
                data = split_data

            if not (data is None or endpoints is None):
                overall_loss += loss(endpoints, data)
        self.logger.add_scalar('losses/weighted/overall', torch.mean(overall_loss))
        return overall_loss


class DynamicFocalLossModule(WeightModule):
    """According to "A Coarse-to-fine Pyramidal Model for Person Re-identification
    via Multi-Loss Dynamic Training" by Zheng et al.
    https://arxiv.org/pdf/1810.12193.pdf
    """

    def __init__(self, delta, tr_loss, id_loss):
        """
        Args:
            tr_loss tuple(name, DynamicFocalLoss): batch hard loss
            id_loss tuple(name, DynamicFocalLoss): softmax loss
        """
        # TODO
        super().__init__({})
        # name is name of dataset
        self.tr_name, self.tr_loss = tr_loss
        self.id_name, self.id_loss = id_loss
        self.delta = delta
        self.tensorboard_logger = logger.get_tensorboard_logger()

    def forward(self, split_endpoints, split_data):
        loss_tr, fl_tr = self.tr_loss(split_endpoints[self.tr_name],
                                      split_data[self.tr_name])
        loss_id, fl_id = self.id_loss(split_endpoints[self.id_name],
                                      split_data[self.id_name])
        if fl_id == 0.0:
            delta = 999
        else:
            delta = fl_tr / fl_id
        self.tensorboard_logger.add_scalar("dynamic_focal/delta", delta)

        # if id loss dominates, only id loss
        if delta < self.delta:
            loss = loss_id
        else:
            loss = fl_tr * loss_tr + fl_id * loss_id

        self.logger.add_scalar('losses/dynamic_focal/overall', torch.mean(loss))
        return loss


class WeightedLoss(nn.Module):
    def __init__(self, loss_module):
        super().__init__()
        self.loss_module = loss_module


class UncertaintyLoss(WeightedLoss):
    """According to "Multi-Task Learning Using Uncertainty to Weight Losses
    for Scene Geometry and Semantics" by Kendall et al."""

    def __init__(self, loss_module, name, init):
        """
        losses (dic): A name-> loss dictionary.
        init (float): 0 -> factor is 1
        """
        super().__init__(loss_module)

        #log(o^2)
        tensor = torch.tensor(init, requires_grad=True)
        self.log_var = torch.nn.Parameter(tensor, requires_grad=True)
        self.name = name
        self.logger = logger.get_tensorboard_logger()

    def forward(self, endpoints, data):
        # 1/o^2
        precision = torch.exp(-self.log_var)
        reg = 0.5 * self.log_var
        self.logger.add_scalar('losses/uncertainty/{}'.format(self.name), precision)
        self.logger.add_scalar('losses/uncertainty/{}/regularization'.format(self.name), reg)
        return precision * self.loss_module(endpoints, data) + reg


class UncertaintyLossNotNegative(WeightedLoss):
    """According to "Auxiliary Tasks in Multi-Task Learning" by Liebel and Koerner."""

    def __init__(self, loss_module, name, init=2.0):
        """
        losses (dic): A name-> loss dictionary.
        init (float): 0 -> factor is 1
        """
        super().__init__(loss_module)

        #log(o^2 + 1)
        init = torch.log(torch.tensor(init))
        tensor = torch.tensor(init, requires_grad=True)
        self.log_var = torch.nn.Parameter(tensor, requires_grad=True)
        self.name = name
        self.logger = logger.get_tensorboard_logger()

    def forward(self, endpoints, data):
        # 1/o^2
        precision = 1 / (torch.exp(self.log_var) - 1)
        # log(sqrt(o^2 + 1))
        reg = 0.5 * self.log_var
        self.logger.add_scalar('losses/uncertainty/{}/precision'.format(self.name), precision)
        self.logger.add_scalar('losses/uncertainty/{}/regularization'.format(self.name), reg)
        return precision * self.loss_module(endpoints, data) + reg


class DynamicFocalKeyLoss(WeightedLoss):
    """According to "Dynamic Task Prioritization for Multitask Learning"
    by Michelle Guo et al."""
    def __init__(self, alpha, gamma, loss_module, name):
        super().__init__(loss_module)
        self.alpha = alpha
        self.gamma = gamma
        self.k = None
        self.tensorboard_logger = logger.get_tensorboard_logger()
        self.name = name

    def forward(self, endpoints, data):
        loss, k = self.loss_module(endpoints, data)
        # TODO make clean
        loss = torch.mean(loss)
        with torch.no_grad():
            k = torch.tensor(k).to(loss.device)
            k = torch.mean(k)
            # just calculate factor no gradient
            if self.k is None:
                # first iteration
                self.k = k
            else:
                old_k = self.k
                self.k = self.alpha * k + (1 - self.alpha) * old_k
                # p is the probability of the occurence of this loss
            fl = -(1-k)**self.gamma * torch.log(k)
        tag = "losses/dynamic_focal/{}/{}"
        self.tensorboard_logger.add_scalar(tag.format(self.name, "k"), self.k)
        self.tensorboard_logger.add_scalar(tag.format(self.name, "fl"), fl)
        return fl * loss


class DynamicFocalLoss(WeightedLoss):
    def __init__(self, alpha, gamma, p0, loss_module, name):
        super().__init__(loss_module)
        self.alpha = alpha
        self.gamma = gamma
        self.k = None
        self.p0 = torch.tensor(p0)
        self.tensorboard_logger = logger.get_tensorboard_logger()
        self.name = name

    def forward(self, endpoints, data):
        loss = self.loss_module(endpoints, data)
        # TODO make clean
        loss = torch.mean(loss)
        with torch.no_grad():
            # just calculate factor no gradient
            if self.k is None:
                # first iteration
                self.k = loss
                p = self.p0
            else:
                old_k = self.k
                self.k = self.alpha * loss + (1 - self.alpha) * old_k
                # p is the probability of the occurence of this loss
                p = torch.min(self.k, old_k) / old_k
            fl = -(1-p)**self.gamma * torch.log(p)
        tag = "losses/dynamic_focal/{}/{}"
        self.tensorboard_logger.add_scalar(tag.format(self.name, "p"), p)
        self.tensorboard_logger.add_scalar(tag.format(self.name, "k"), self.k)
        self.tensorboard_logger.add_scalar(tag.format(self.name, "fl"), fl)
        return loss, fl


class LinearWeightedLoss(WeightedLoss):
    def __init__(self, weight, loss_module):
        super().__init__(loss_module)
        self.weight = weight

    def forward(self, endpoints, data):
        loss = self.loss_module(endpoints, data)
        return self.weight * loss
    
