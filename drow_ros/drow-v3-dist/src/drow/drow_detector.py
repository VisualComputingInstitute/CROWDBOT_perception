import torch
import torch.nn.functional as F

from model.drow import FastDROWNet3LF2p
import utils.utils as u


class DROWDetector(object):
    def __init__(self, ckpt_file, num_scans=5, num_cutout_pts=48, gpu=0,
                 sequential_inference=True):
        self._gpu, self._num_cutout_pts = gpu, num_cutout_pts
        self._laser_angle = None

        # net
        model = FastDROWNet3LF2p(num_scans=num_scans,
                                 sequential_inference=sequential_inference)
        ckpt = torch.load(ckpt_file)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        self._model = model.cuda() if self._gpu is not None else model

    def __call__(self, scan):
        assert self._laser_angle is not None

        # generate network input
        scan = scan[None, ...]  # Expand one dimension for sequential scans
        angle_incre = self._laser_angle[1] - self._laser_angle[0]
        cutout = u.scans_to_cutout(scans=scan,
                                   angle_incre=angle_incre,
                                   num_cutout_pts=self._num_cutout_pts)
        cutout = cutout[None, ...]  # Expend one dimension for batch

        if self._gpu is not None:
            cutout = torch.from_numpy(cutout).cuda(non_blocking=True).float()

        # inference
        with torch.no_grad():
            pred_cls, pred_reg = self._model(cutout)
            pred_cls = F.softmax(pred_cls, dim=-1).data.cpu().numpy()
            pred_reg = pred_reg.data.cpu().numpy()

        # post processing
        dets_xy, dets_cls = u.group_predicted_center(
                scan[0], self._laser_angle, pred_cls[0], pred_reg[0])

        return dets_xy, dets_cls

    def set_laser_spec(self, angle_inc, num_pts):
        self._laser_angle = u.get_laser_phi(angle_inc, num_pts)

    def laser_spec_set(self):
        return self._laser_angle is not None