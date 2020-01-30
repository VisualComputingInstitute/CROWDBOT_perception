import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import utils.utils as u
import utils.v2_utils as v2u
from utils.train_utils import load_checkpoint
from model.drow import FastDROWNet3LF2p


def create_model(num_scans, ckpt_file, v2_format=False):
    model = FastDROWNet3LF2p(num_scans=5)
    if v2_format:
       load = torch.load(ckpt_file)
       model.load_state_dict(load['model'])
    else:
        load_checkpoint(model=model, filename=ckpt_file)

    return model.cuda()


def eval_model(model, test_loader, output_file=None):
    model.eval()

    # detection location
    dets_list = []
    dets_cls_list = []
    dets_ns_list = []

    # ground truth xy of different class
    gts_dict = {}
    gts_dict['dets_wc'] = []
    gts_dict['dets_wa'] = []
    gts_dict['dets_wp'] = []
    gts_dict['dets_all'] = []

    # ground truth frame stamp
    gts_ns_dict = {}
    gts_ns_dict['dets_wc'] = []
    gts_ns_dict['dets_wa'] = []
    gts_ns_dict['dets_wp'] = []
    gts_ns_dict['dets_all'] = []

    # inference over the whole test set, and collect results
    for data in tqdm.tqdm(test_loader, desc='test'):
        # inference
        cutout = torch.from_numpy(data['cutout']).float().cuda()
        pred_cls, pred_reg = model(cutout)
        pred_cls = F.softmax(pred_cls, dim=-1).data.cpu().numpy()[0]
        pred_reg = pred_reg.data.cpu().numpy()[0]

        # parse and collect detection
        scan = data['scans'][0][-1, :]
        dets_xs, dets_ys, dets_cls = u.group_predicted_center(scan,
                                                              u.get_laser_phi(),
                                                              pred_cls,
                                                              pred_reg)
        dets_ns = data['dets_ns'][0]
        for x, y, c in zip(dets_xs, dets_ys, dets_cls):
            dets_list.append((x, y))
            dets_cls_list.append(c)
            dets_ns_list.append(dets_ns)

        # parse and collect ground truth
        for k in ['dets_wc', 'dets_wa', 'dets_wp']:
            if len(data[k][0]) > 0:
                for r, phi in data[k][0]:
                    xy = u.rphi_to_xy(r, phi)
                    gts_dict[k].append(xy)
                    gts_ns_dict[k].append(dets_ns)
                    gts_dict['dets_all'].append(xy)
                    gts_ns_dict['dets_all'].append(dets_ns)

    dets_xy = np.array(dets_list)
    dets_cls = np.array(dets_cls_list)
    dets_ns = np.array(dets_ns_list)

    for k, v in gts_dict.items():
        gts_dict[k] = np.array(v)

    for k, v in gts_ns_dict.items():
        gts_ns_dict[k] = np.array(v)

    # compute pr curve
    rpt_dict = {}  # rpt = (recall, precision, threshold)
    eval_radius = 0.5
    rpt_dict['wc'] = v2u.prec_rec_2d(
            dets_cls[:, 1], dets_xy, dets_ns,
            gts_dict['dets_wc'], gts_ns_dict['dets_wc'],
            np.full_like(gts_ns_dict['dets_wc'], eval_radius, dtype=np.float32))

    rpt_dict['wa'] = v2u.prec_rec_2d(
            dets_cls[:, 2], dets_xy, dets_ns,
            gts_dict['dets_wa'], gts_ns_dict['dets_wa'],
            np.full_like(gts_ns_dict['dets_wa'], eval_radius, dtype=np.float32))

    rpt_dict['wp'] = v2u.prec_rec_2d(
            dets_cls[:, 3], dets_xy, dets_ns,
            gts_dict['dets_wp'], gts_ns_dict['dets_wp'],
            np.full_like(gts_ns_dict['dets_wp'], eval_radius, dtype=np.float32))

    rpt_dict['all'] = v2u.prec_rec_2d(
            np.sum(dets_cls[:, 1:], axis=1), dets_xy, dets_ns,
            gts_dict['dets_all'], gts_ns_dict['dets_all'],
            np.full_like(gts_ns_dict['dets_all'], eval_radius, dtype=np.float32))

    # save result
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(rpt_dict, f)

    return rpt_dict


def plot_eval_result(rpt_dict, plot_title=None, output_file=None):
    fig, ax = v2u.plot_prec_rec(wds=rpt_dict['all'],
                                wcs=rpt_dict['wc'],
                                was=rpt_dict['wa'],
                                wps=rpt_dict['wp'],
                                title=plot_title)

    if output_file is not None:
        plt.savefig(output_file, bbox_inches='tight')

    return fig, ax
