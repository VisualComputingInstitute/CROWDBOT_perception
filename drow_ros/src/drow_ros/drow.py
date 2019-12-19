from collections import deque

import numpy as np
from scipy import ndimage
import cv2

import torch
import torch.nn.functional as F

from model import DROWNet3LF2p


torch.backends.cudnn.benchmark = True  # Run benchmark to select fastest implementation of ops.

def rphi_to_xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan_to_xy(scan, thresh=104.0):
    s = scan
    if thresh is not None:
        s[s > thresh] = np.nan
    angles = np.linspace(-225.0 / 2, 225.0 / 2, len(scan)) / 180.0 * np.pi
    return rphi_to_xy(s, angles)


def pred_offset_to_xy(scan, pred_offsets, angle_grid):
    """
    @brief      Convert predicted offset from network to xy coordinate in
                laser frame.

    @return     (numpy.array, numpy.array) x, y as 1D array.
    """
    r_parallel = scan + pred_offsets[:, 1]
    r = np.hypot(r_parallel, pred_offsets[:, 0])
    # r = np.sqrt(np.square(r_parallel) + np.square(pred_offsets[:, 0]))
    phi = angle_grid - np.arctan2(pred_offsets[:, 0], r_parallel)
    return rphi_to_xy(r, phi)


def scans_to_cutouts(scans, angle_increment, window_width=1.66, n_points=48,
                     dist_thres=1.0, pad_val=29.99):
    # UNK = 29.99
    # window_width = 1.66
    # thresh_dist = 1
    # number_cutout_points = 48
    # number_scans, number_points = scans.shape
    n_scans, s_size = scans.shape
    cutouts = np.empty((s_size, n_scans, n_points), dtype=np.float32)
    scans = np.pad(scans, ((0, 0), (0, 1)), mode='constant', constant_values=pad_val)

    # Compute the size (angle) of the window
    for idx_p in range(s_size):
        center_z = scans[-1, idx_p] + 1e-6  # Prevent dividing by zero
        half_alpha = float(np.arctan(0.5 * window_width / center_z))

        for idx_s in range(n_scans):
            # Compute the start and end indices of points in the scan to be
            # considered.
            idx0 = int(round(idx_p - half_alpha / angle_increment))
            idx1 = int(round(idx_p + half_alpha / angle_increment))
            p_indices = np.arange(idx0, idx1 + 1)
            p_indices.clip(-1, s_size, out=p_indices)
            # Write the scan into the buffer which has UNK at the end and
            # then sample from it.
            cutout = scans[idx_s, p_indices]

            # Interpolate (downsample) datapoint
            interp = cv2.INTER_AREA if n_points < len(cutout) else cv2.INTER_LINEAR
            cutout = cv2.resize(cutout, (1, n_points),
                                interpolation=interp).squeeze()

            # Clip things too close and too far to create the "focus tunnel"
            # since they are likely irrelevant.
            cutout.clip(center_z - dist_thres, center_z + dist_thres, out=cutout)
            # center_z = scans[idx_s, idx_p]
            cutout -= center_z
            cutouts[idx_p, idx_s, :] = cutout

    return cutouts


def prediction_to_detection(xs, ys, probas, class_weights=None,
                            x_min=-25, x_max=25, y_min=-25, y_max=25,
                            bin_size=0.5, vote_collect_radius=0.8,
                            min_thresh=1e-5,
                            blur_size=21, blur_sigma=2.0,
                            nms_size=3):
    # Apply class weights.
    if class_weights is not None:
        probas = np.array(probas)  # Make a copy.
        probas[:, 1:] *= class_weights

    # Create voting grid.
    x_number_bins = int((x_max - x_min) / bin_size)
    y_number_bins = int((y_max - y_min) / bin_size)
    x_max = x_min + x_number_bins * bin_size
    y_max = y_min + y_number_bins * bin_size
    votes_grid = np.zeros((x_number_bins, y_number_bins, probas.shape[1]), np.float32)

    # Remove out-of-range prediction.
    in_range = np.all([xs >= x_min, xs < x_max, ys >= y_min, ys < y_max], axis=0)
    xs, ys, probas = xs[in_range], ys[in_range], probas[in_range]

    # Remove wake prediction.
    is_object = np.sum(probas[:, 1:], axis=-1) > min_thresh
    xs, ys, probas = xs[is_object], ys[is_object], probas[is_object]

    if len(xs) == 0:
        return

    # Collect votes.
    xs_idx = ((xs - x_min) / bin_size).astype(np.int32)
    ys_idx = ((ys - y_min) / bin_size).astype(np.int32)
    votes = np.concatenate(
            (np.sum(probas[:, 1:], axis=-1, keepdims=True), probas[:, 1:]),
            axis=-1)
    np.add.at(votes_grid, (xs_idx, ys_idx), votes)

    # NMS based on objectiveness votes.
    obj_votes_grid = votes_grid[..., 0]
    if blur_size > 0 and blur_sigma > 0:
        cv2.GaussianBlur(obj_votes_grid, (blur_size, blur_size), blur_sigma)
    obj_votes_max = ndimage.maximum_filter(obj_votes_grid, size=nms_size)
    is_max = np.logical_and(obj_votes_grid > 0.0, obj_votes_grid == obj_votes_max)
    max_xs_idx, max_ys_idx = np.where(is_max)

    number_dets = len(max_xs_idx)
    if number_dets == 0:
        return

    # Associate each prediction with peaks on objectiveness.
    obj_xs = max_xs_idx * bin_size + x_min + bin_size / 2.0
    obj_ys = max_ys_idx * bin_size + y_min + bin_size / 2.0
    dist_to_obj_sq = np.square(xs - obj_xs[:, np.newaxis]) \
            + np.square(ys - obj_ys[:, np.newaxis])
    pred_id = np.argmin(dist_to_obj_sq, axis=0)

    # Combine all points associate to same objectiveness peaks and vote for
    # predicted center and probability
    xs_det = np.empty(number_dets, dtype=np.float32)
    ys_det = np.empty(number_dets, dtype=np.float32)
    probas_det = np.empty((number_dets, probas.shape[1]), dtype=np.float32)
    vote_collect_radius_sq = vote_collect_radius * vote_collect_radius
    for i_det in range(number_dets):
        voters_idx = np.where(pred_id == i_det)[0]
        is_valid = dist_to_obj_sq[i_det, voters_idx] < vote_collect_radius_sq
        voters_idx = voters_idx[is_valid]
        xs_det[i_det] = np.mean(xs[voters_idx])
        ys_det[i_det] = np.mean(ys[voters_idx])
        probas_det[i_det] = np.mean(probas[voters_idx], axis=0)

    return xs_det, ys_det, probas_det


class Detector(object):
    def __init__(self, f_checkpoint, ntime=5, nsamp=48, gpu=0):
        self._ntime = ntime
        self._nsamp = nsamp

        self._gpu = gpu  # This is the GPU index, use `False` for CPU-only.

        self._net = DROWNet3LF2p(self._ntime, 2.5)
        if gpu is not False and torch.cuda.is_available():
            self._net = self._net.cuda(device=self._gpu)

        load = torch.load(f_checkpoint)
        self._net.load_state_dict(load['model'])
        self._net.eval()

        self._scans = deque(maxlen=self._ntime)

        self.angle_min = None
        self.angle_max = None
        self.scan_points = None
        self.angle_increment = None
        self.angle_grid = None

    def init(self, angle_min=-1.9634954084936207, angle_max=1.9634954084936207,
             scan_points=450):
        self.angle_min = angle_min
        self.angle_max = angle_max
        self.scan_points = scan_points
        self.angle_increment = (self.angle_max - self.angle_min) / self.scan_points
        self.angle_grid = np.linspace(self.angle_min, self.angle_max, self.scan_points)

    def initialized(self):
        return self.angle_grid is not None and self.angle_increment is not None

    def __call__(self, scan):
        self._scans.append(scan)
        xb = self._get_input_tensor()

        with torch.no_grad():
            logits, pred_offsets = self._net(xb)
            pred_probs = F.softmax(logits, dim=-1).data.cpu().numpy()
            pred_offsets = pred_offsets.data.cpu().numpy()

        pred_xs, pred_ys = pred_offset_to_xy(scan, pred_offsets, self.angle_grid)

        return pred_xs, pred_ys, pred_probs

    def _get_input_tensor(self):
        """
        @brief      Form input tensor to the network.
        """
        scans = np.array(self._scans)
        # Prepend the exact same scan/odom for the first few where there's no history.
        if len(self._scans) != self._ntime:
            pad_number = self._ntime - len(self._scans)
            pad = np.tile(scans[0], pad_number).reshape(pad_number, -1)
            scans = np.concatenate((pad, scans), axis=0)

        xb = torch.from_numpy(scans_to_cutouts(
                scans, self.angle_increment, n_points=self._nsamp))

        if self._gpu is not False and torch.cuda.is_available():
            xb = xb.cuda(device=self._gpu)

        return xb


if __name__=='__main__':
    import matplotlib.pyplot as plt

    d = Detector()
    d.init()
    scan_file = '/home/jia/git/drow-ros/drow_ros/bags.csv'
    scans = np.genfromtxt(scan_file, delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for scan in scans:
        scan_x, scan_y = scan_to_xy(scan)
        x, y, conf = d(scan)
        label = np.argmax(conf, axis=1)

        plt.cla()
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        ax.scatter(scan_x, scan_y, s=1)
        ax.scatter(x[label==1], y[label==1], s=1, c='blue')  # Wheel chair?
        ax.scatter(x[label==2], y[label==2], s=1, c='green')  # Pedestrian?
        ax.scatter(x[label==3], y[label==3], s=1, c='brown')  # Walker?
        plt.pause(0.1)

    plt.show()