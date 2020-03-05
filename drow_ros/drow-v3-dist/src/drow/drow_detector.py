import os
import yaml
import torch
import utils.utils as u

import rospkg


def cfg_to_model(cfg):
    if cfg['network'] == 'cutout':
        from model.drow import DROW
        model = DROW(num_scans=cfg['num_scans'],
                     num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                     focal_loss_gamma=cfg['focal_loss_gamma'],
                     pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'cutout_gating':
        from model.drow import TemporalDROW
        model = TemporalDROW(num_scans=cfg['num_scans'],
                             num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                             focal_loss_gamma=cfg['focal_loss_gamma'],
                             pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'cutout_spatial':
        from model.drow import SpatialDROW
        model = SpatialDROW(num_scans=cfg['num_scans'],
                            num_pts=cfg['cutout_kwargs']['num_cutout_pts'],
                            focal_loss_gamma=cfg['focal_loss_gamma'],
                            alpha=cfg['similarity_kwargs']['alpha'],
                            window_size=cfg['similarity_kwargs']['window_size'],
                            pedestrian_only=cfg['pedestrian_only'])

    elif cfg['network'] == 'fc2d':
        from model.polar_drow import PolarDROW
        model = PolarDROW(in_channel=1)

    elif cfg['network'] == 'fc2d_fea':
        raise NotImplementedError
        from model.polar_drow import PolarDROW
        model = PolarDROW(in_channel=cfg['cutout_kwargs']['num_cutout_pts'])

    elif cfg['network'] == 'fc1d':
        from model.fconv_drow import FConvDROW
        model = FConvDROW(in_channel=1)

    elif cfg['network'] == 'fc1d_fea':
        from model.fconv_drow import FConvDROW
        model = FConvDROW(in_channel=cfg['cutout_kwargs']['num_cutout_pts'])

    else:
        raise RuntimeError

    return model


class DROWDetector(object):
    def __init__(self, ckpt_file, use_spaam=False):
        self._laser_angle = None

        rospack = rospkg.RosPack()
        cfg_dir = os.path.join(rospack.get_path('drow'), 'src/drow/cfgs')
        cfg = os.path.join(cfg_dir, 'SPA_11_5.yaml') if use_spaam \
                else os.path.join(cfg_dir, 'PSG_small_width_depth_56.yaml')

        with open(cfg, 'r') as f:
            self._cfg = yaml.safe_load(f)

        self._model = cfg_to_model(self._cfg)
        self._model.cuda()

        ckpt = torch.load(ckpt_file)
        self._model.load_state_dict(ckpt['model_state'])
        self._model.eval()

    def __call__(self, scan):
        assert self._laser_angle is not None

        angle_incre = self._laser_angle[1] - self._laser_angle[0]
        cutout = u.scans_to_cutout(scans=scan[None, ...],
                                   angle_incre=angle_incre,
                                   **self._cfg['cutout_kwargs'])

        cutout = torch.from_numpy(cutout[None, ...]).cuda(non_blocking=True).float()

        # inference
        with torch.no_grad():
            pred_cls, pred_reg = self._model(cutout)
            pred_cls = torch.sigmoid(pred_cls[0]).data.cpu().numpy()
            pred_reg = pred_reg[0].data.cpu().numpy()

        # post processing
        dets_xy, dets_cls, _ = u.group_predicted_center(
                scan, self._laser_angle, pred_cls, pred_reg)

        return dets_xy, dets_cls

    def set_laser_spec(self, angle_inc, num_pts, stride=1):
        self._laser_angle = u.get_laser_phi(angle_inc, num_pts)[::stride]

    def laser_spec_set(self):
        return self._laser_angle is not None




