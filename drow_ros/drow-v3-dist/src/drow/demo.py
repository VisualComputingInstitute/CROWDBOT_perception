import argparse
import matplotlib.pyplot as plt
import numpy as np
import yaml

import torch
import torch.nn.functional as F

import utils.eval_utils as eu
import utils.utils as u
from utils.dataset import create_test_dataloader


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument("--cfg", type=str, required=True, help="configuration of the experiment")
parser.add_argument("--ckpt", type=str, required=True)
args = parser.parse_args()

with open(args.cfg, 'r') as f:
    cfg = yaml.safe_load(f)

conf_thresh = 0.2
test_loader = create_test_dataloader(data_path="../data/DROWv2-data",
                                     num_scans=cfg['num_scans'],
                                     use_polar_grid=cfg['use_polar_grid'],
                                     cutout_kwargs=cfg['cutout_kwargs'],
                                     polar_grid_kwargs=cfg['polar_grid_kwargs'])
model = eu.create_model(args.ckpt, num_scans=cfg['num_scans'])
model.eval()
model = model.cuda()

fig = plt.figure()
ax = fig.add_subplot(111)

for data in test_loader:
    # inference
    cutout = torch.from_numpy(data['cutout']).float().cuda()
    with torch.set_grad_enabled(False):
        pred_cls, pred_reg = model(cutout)
        pred_cls = F.softmax(pred_cls, dim=2)
    pred_cls = pred_cls.cpu().numpy()[0]
    pred_reg = pred_reg.cpu().numpy()[0]

    # group prediction
    scan = data['scans'][0][-1, :]
    dets_xs, dets_ys, dets_cls = u.group_predicted_center(scan,
                                                          u.get_laser_phi(),
                                                          pred_cls,
                                                          pred_reg)

    # confidence threshold
    dets_cls_label = np.argmax(dets_cls[:, 1:], axis=1) + 1
    dets_cls_conf = np.max(dets_cls[:, 1:], axis=1)
    dets_cls_label[dets_cls_conf < conf_thresh] = 0

    # target
    target_cls, target_reg = data['target_cls'][0], data['target_reg'][0]

    scan_phi = u.get_laser_phi()
    scan_x, scan_y = u.scan_to_xy(scan)

    # plot
    plt.cla()
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    cls_labels = [1, 2, 3]
    cls_colors = ['blue', 'green', 'red']

    # points and annotation
    ax.scatter(scan_x, scan_y, s=1, c='black')
    for cls_label, c in zip(cls_labels, cls_colors):
        canonical_dxy = target_reg[target_cls==cls_label]
        gt_r, gt_phi = u.canonical_to_global(
                scan[target_cls==cls_label],
                scan_phi[target_cls==cls_label],
                canonical_dxy[:, 0],
                canonical_dxy[:, 1])
        gt_xs, gt_ys = u.rphi_to_xy(gt_r, gt_phi)
        ax.scatter(gt_xs, gt_ys, s=25, c=c)

    # network result
    for cls_label, c in zip(cls_labels, cls_colors):
        ax.scatter(dets_xs[dets_cls_label==cls_label],
                   dets_ys[dets_cls_label==cls_label],
                   s=75, c=c, marker="+")

    plt.pause(0.1)
