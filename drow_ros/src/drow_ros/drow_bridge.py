from collections import deque

import numpy as np
from scipy import ndimage
import cv2




from model import DROWNet3LF2p


class DROWBridge(object):
    def __init__(self, drow_path, ckpt_path):














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