from joblib import Parallel, delayed
import numpy as np
from scipy.ndimage import maximum_filter
from scipy.spatial.distance import cdist
import cv2

# In numpy >= 1.17, np.clip is slow, use core.umath.clip instead
# https://github.com/numpy/numpy/issues/14281
if "clip" in dir(np.core.umath):
    _clip = np.core.umath.clip
    # print("use np.core.umath.clip")
else:
    _clip = np.clip
    # print("use np.clip")

def get_laser_phi(angle_inc=np.radians(0.5), num_pts=450):
    # Default setting of DROW, which use SICK S300 laser, with 225 deg fov
    # and 450 pts, mounted at 37cm height.
    laser_fov = (num_pts - 1) * angle_inc  # 450 points
    return np.linspace(-laser_fov*0.5, laser_fov*0.5, num_pts)


def scan_to_xy(scan, phi=None):
    if phi is None:
        return rphi_to_xy(scan, get_laser_phi())
    else:
        return rphi_to_xy(scan, phi)


def xy_to_rphi(x, y):
    # NOTE: Axes rotated by 90 CCW by intent, so that 0 is top.
    # y axis aligns with the center of scan, pointing outward/upward, x axis pointing to right
    # phi is the angle with y axis, rotating towards x is positive
    return np.hypot(x, y), np.arctan2(x, y)


def rphi_to_xy(r, phi):
    return r * np.sin(phi), r * np.cos(phi)


def global_to_canonical(scan_r, scan_phi, dets_r, dets_phi):
    # Canonical frame: origin at the scan points, y pointing outward/upward along the scan, x pointing rightward
    dx = np.sin(dets_phi - scan_phi) * dets_r
    dy = np.cos(dets_phi - scan_phi) * dets_r - scan_r
    return dx, dy


def canonical_to_global(scan_r, scan_phi, dx, dy):
    tmp_y = scan_r + dy
    tmp_phi = np.arctan2(dx, tmp_y)  # dx first is correct due to problem geometry dx -> y axis and vice versa.
    dets_phi = tmp_phi + scan_phi
    dets_r = tmp_y / np.cos(tmp_phi)
    return dets_r, dets_phi


def data_augmentation(sample_dict):
    scans, target_reg = sample_dict['scans'], sample_dict['target_reg']

    # # Random scaling
    # s = np.random.uniform(low=0.95, high=1.05)
    # scans = s * scans
    # target_reg = s * target_reg

    # Random left-right flip. Of whole batch for convenience, but should be the same as individuals.
    if np.random.rand() < 0.5:
        scans = scans[:, ::-1]
        target_reg[:, 0] = -target_reg[:, 0]

    sample_dict.update({'target_reg': target_reg, 'scans': scans})

    return sample_dict


def get_regression_target(scan, scan_phi, wcs, was, wps,
                          radius_wc=0.6, radius_wa=0.4, radius_wp=0.35,
                          label_wc=1, label_wa=2, label_wp=3,
                          pedestrian_only=False):
    num_pts = len(scan)
    target_cls = np.zeros(num_pts, dtype=np.int64)
    target_reg = np.zeros((num_pts, 2), dtype=np.float32)

    if pedestrian_only:
        all_dets = list(wps)
        all_radius = [radius_wp] * len(wps)
        labels = [0] + [1] * len(wps)
    else:
        all_dets = list(wcs) + list(was) + list(wps)
        all_radius = [radius_wc]*len(wcs) + [radius_wa]*len(was) + [radius_wp]*len(wps)
        labels = [0] + [label_wc] * len(wcs) + [label_wa] * len(was) + [label_wp] * len(wps)

    dets = closest_detection(scan, scan_phi, all_dets, all_radius)

    for i, (r, phi) in enumerate(zip(scan, scan_phi)):
        if 0 < dets[i]:
            target_cls[i] = labels[dets[i]]
            target_reg[i,:] = global_to_canonical(r, phi, *all_dets[dets[i]-1])

    return target_cls, target_reg


def closest_detection(scan, scan_phi, dets, radii):
    """
    Given a single `scan` (450 floats), a list of r,phi detections `dets` (Nx2),
    and a list of N `radii` for those detections, return a mapping from each
    point in `scan` to the closest detection for which the point falls inside its radius.
    The returned detection-index is a 1-based index, with 0 meaning no detection
    is close enough to that point.
    """
    if len(dets) == 0:
        return np.zeros_like(scan, dtype=int)

    assert len(dets) == len(radii), "Need to give a radius for each detection!"

    # Distance (in x,y space) of each laser-point with each detection.
    scan_xy = np.array(rphi_to_xy(scan, scan_phi)).T  # (N, 2)
    dists = cdist(scan_xy, np.array([rphi_to_xy(r, phi) for r, phi in dets]))

    # Subtract the radius from the distances, such that they are < 0 if inside, > 0 if outside.
    dists -= radii

    # Prepend zeros so that argmin is 0 for everything "outside".
    dists = np.hstack([np.zeros((len(scan), 1)), dists])

    # And find out who's closest, including the threshold!
    return np.argmin(dists, axis=1)


def scans_to_cutout_vectorize(scans, angle_incre, fixed=True, centered=True, pt_inds=None,
                              window_width=1.66, window_depth=1.0, num_cutout_pts=48,
                              padding_val=29.99):
    """ TODO: Probably we can still try to clean this up more.
    This function here only creates a single cut-out; for training,
    we'd want to get a batch of cutouts from each seq (can vectorize) and for testing
    we'd want all cutouts for one scan, which we can vectorize too.
    But ain't got time for this shit!

    Args:
    - scans: (T,N) the T scans (of scansize N) to cut out from, `T=0` being the "current time".
    - out: None or a (T,nsamp) buffer where to store the cutouts.
    """
    # Don't use this version, slow and has bug
    raise NotImplementedError

    num_scans, num_pts = scans.shape
    if pt_inds is None:
        pt_inds = np.arange(num_pts)

    # size (width) of the window
    pt_r = scans[:, pt_inds] if fixed else np.tile(scans[-1, pt_inds], num_scans).reshape(num_scans, -1)
    half_alpha = np.arctan(0.5 * window_width / pt_r.clip(min=1e-2))
    half_win_inds = np.ceil(half_alpha / angle_incre)

    # start and end indices of cutout
    scans_padded = np.pad(scans, ((0, 0), (1, 1)), mode='constant', constant_values=padding_val)  # pad boarder
    start_inds = (pt_inds - half_win_inds + 1).clip(min=0).astype(np.int)  # (num_scan, len(pt_inds)), +1 for padding
    end_inds = (pt_inds + half_win_inds + 2).clip(max=num_pts+1).astype(np.int)

    # do cutout
    scans_cutout = np.empty((len(pt_inds), num_scans, num_cutout_pts), dtype=np.float32)
    for idx_pt, (inds_0, inds_1) in enumerate(zip(start_inds.T, end_inds.T)):
        for idx_scan, (idx_0, idx_1) in enumerate(zip(inds_0, inds_1)):
            # neighbor points
            ct_pts = scans_padded[idx_scan, idx_0:idx_1]

            # resampling
            interp = cv2.INTER_AREA if num_cutout_pts < len(ct_pts) else cv2.INTER_LINEAR
            ct_sampled = cv2.resize(ct_pts,
                                    (1, num_cutout_pts),
                                    interpolation=interp).squeeze()

            scans_cutout[idx_pt, idx_scan, :] = ct_sampled

    # clip depth to avoid strong depth discontinuity
    scans_cutout = scans_cutout.clip(min=pt_r.T - window_depth, max=pt_r.T + window_depth)  # clip

    # normalize
    if centered:
        scans_cutout -= pt_r.T  # center
        scans_cutout = scans_cutout / window_depth  # normalize

    return scans_cutout


def scans_to_cutout(scans, angle_incre, fixed=True, centered=True, pt_inds=None,
                    window_width=1.66, window_depth=1.0, num_cutout_pts=48,
                    padding_val=29.99):
    """ TODO: Probably we can still try to clean this up more.
    This function here only creates a single cut-out; for training,
    we'd want to get a batch of cutouts from each seq (can vectorize) and for testing
    we'd want all cutouts for one scan, which we can vectorize too.
    But ain't got time for this shit!

    Args:
    - scans: (T,N) the T scans (of scansize N) to cut out from, `T=0` being the "current time".
    - out: None or a (T,nsamp) buffer where to store the cutouts.
    """
    num_scans, num_pts = scans.shape
    if pt_inds is None:
        pt_inds = range(num_pts)

    scans_padded = np.pad(scans, ((0, 0), (0, 1)), mode='constant', constant_values=padding_val)  # pad boarder
    scans_cutout = np.empty((num_pts, num_scans, num_cutout_pts), dtype=np.float32)

    for scan_idx in range(num_scans):
        for pt_idx in pt_inds:
            # Compute the size (width) of the window
            pt_r = scans[scan_idx, pt_idx] if fixed else scans[-1, pt_idx]

            half_alpha = float(np.arctan(0.5 * window_width / max(pt_r, 0.01)))

            # Compute the start and end indices of cutout
            start_idx = int(round(pt_idx - half_alpha / angle_incre))
            end_idx = int(round(pt_idx + half_alpha / angle_incre))
            cutout_pts_inds = np.arange(start_idx, end_idx + 1)
            cutout_pts_inds = _clip(cutout_pts_inds, -1, num_pts)
            # cutout_pts_inds = np.core.umath.clip(cutout_pts_inds, -1, num_pts)
            # cutout_pts_inds = cutout_pts_inds.clip(-1, num_pts)

            # cutout points
            cutout_pts = scans_padded[scan_idx, cutout_pts_inds]

            # resampling/interpolation
            interp = cv2.INTER_AREA if num_cutout_pts < len(cutout_pts_inds) else cv2.INTER_LINEAR
            cutout_sampled = cv2.resize(cutout_pts,
                                       (1, num_cutout_pts),
                                       interpolation=interp).squeeze()

            # center cutout and clip depth to avoid strong depth discontinuity
            cutout_sampled = _clip(cutout_sampled, pt_r - window_depth, pt_r + window_depth)
            # cutout_sampled = np.core.umath.clip(
            #         cutout_sampled,
            #         pt_r - window_depth, 
            #         pt_r + window_depth)
            # cutout_sampled = cutout_sampled.clip(pt_r - window_depth,
            #                                      pt_r + window_depth)

            if centered:
                cutout_sampled -= pt_r  # center
                cutout_sampled = cutout_sampled / window_depth  # normalize
            scans_cutout[pt_idx, scan_idx, :] = cutout_sampled

    return scans_cutout


def scans_to_cutout_new(scans, scan_phi, output_phi, fixed=True, centered=True,
                        window_width=1.66, window_depth=1.0, num_cutout_pts=48,
                        padding_val=29.99):
    """
    Args:
        scans: np.array (T, N), T scans with N number of points, temporal order
               should be ascending, i.e. scans[-1] should be the latest scan
        scan_phi: np.array (N,)
    """
    num_scans, num_pts = scans.shape
    num_output_pts = len(output_phi)

    scans_padded = np.pad(scans, ((0, 0), (0, 1)), mode='constant', constant_values=padding_val)  # pad boarder
    scans_cutout = np.empty((num_output_pts, num_scans, num_cutout_pts), dtype=np.float32)
    output_scan = np.empty(num_output_pts, dtype=np.float32)
    angle_incre = scan_phi[1] - scan_phi[0]

    for o_pt_idx in range(num_output_pts):
        o_phi = output_phi[o_pt_idx]
        pt_idx = int(round((o_phi - scan_phi[0]) / angle_incre))
        pt_idx = max(min(pt_idx, num_pts - 1), 0)
        output_scan[o_pt_idx] = scans[-1, pt_idx]
        for scan_idx in range(num_scans):
            # angular size (width) of the window
            pt_r = scans[scan_idx, pt_idx] if fixed else scans[-1, pt_idx]
            half_alpha = float(np.arctan(0.5 * window_width / max(pt_r, 0.01)))

            # start and end indices of cutout
            s_angle, e_angle = o_phi - half_alpha, o_phi + half_alpha
            s_idx = int(round((s_angle - scan_phi[0]) / angle_incre))
            e_idx = int(round((e_angle - scan_phi[0]) / angle_incre))
            # s_idx = int(round(pt_idx - half_alpha / angle_incre))
            # e_idx = int(round(pt_idx + half_alpha / angle_incre))
            cutout_pts_inds = np.arange(s_idx, e_idx + 1)
            cutout_pts_inds = cutout_pts_inds.clip(-1, num_pts)

            # cutout points
            cutout_pts = scans_padded[scan_idx, cutout_pts_inds]

            # resampling/interpolation
            interp = cv2.INTER_AREA if num_cutout_pts < len(cutout_pts_inds) else cv2.INTER_LINEAR
            cutout_sampled = cv2.resize(cutout_pts,
                                       (1, num_cutout_pts),
                                       interpolation=interp).squeeze()

            # center cutout and clip depth to avoid strong depth discontinuity
            cutout_sampled = cutout_sampled.clip(pt_r - window_depth,
                                                 pt_r + window_depth)  # clip
            if centered:
                cutout_sampled -= pt_r  # center
                cutout_sampled = cutout_sampled / window_depth  # normalize

            # output
            scans_cutout[o_pt_idx, scan_idx, :] = cutout_sampled

    return scans_cutout, output_scan


def scans_to_cutout_parallel(scans, angle_incre, fixed=True, centered=True, pt_inds=None,
                             window_width=1.66, window_depth=1.0, num_cutout_pts=48,
                             padding_val=29.99):
    """ TODO: Probably we can still try to clean this up more.
    This function here only creates a single cut-out; for training,
    we'd want to get a batch of cutouts from each seq (can vectorize) and for testing
    we'd want all cutouts for one scan, which we can vectorize too.
    But ain't got time for this shit!

    Args:
    - scans: (T,N) the T scans (of scansize N) to cut out from, `T=0` being the "current time".
    - out: None or a (T,nsamp) buffer where to store the cutouts.
    """
    num_scans, num_pts = scans.shape
    if pt_inds is None:
        pt_inds = range(num_pts)

    scans_padded = np.pad(scans, ((0, 0), (0, 1)), mode='constant', constant_values=padding_val)  # pad boarder
    scans_cutout = np.empty((num_pts, num_scans, num_cutout_pts), dtype=np.float32)

    # do cutout
    def _ct_pt(pt_idx):
        for scan_idx in range(num_scans):
            # Compute the size (width) of the window
            pt_r = scans[scan_idx, pt_idx] if fixed else scans[-1, pt_idx]
            half_alpha = float(np.arctan(0.5 * window_width / max(pt_r, 1e-2)))

            # Compute the start and end indices of cutout
            start_idx = int(round(pt_idx - half_alpha / angle_incre))
            end_idx = int(round(pt_idx + half_alpha / angle_incre))
            cutout_pts_inds = np.arange(start_idx, end_idx + 1)
            cutout_pts_inds = cutout_pts_inds.clip(-1, num_pts)

            # cutout points
            cutout_pts = scans_padded[scan_idx, cutout_pts_inds]

            # resampling/interpolation
            interp = cv2.INTER_AREA if num_cutout_pts < len(cutout_pts_inds) else cv2.INTER_LINEAR
            cutout_sampled = cv2.resize(cutout_pts,
                                       (1, num_cutout_pts),
                                       interpolation=interp).squeeze()

            # center cutout and clip depth to avoid strong depth discontinuity
            cutout_sampled = cutout_sampled.clip(pt_r - window_depth,
                                                 pt_r + window_depth)  # clip
            if centered:
                cutout_sampled -= pt_r  # center
                cutout_sampled = cutout_sampled / window_depth  # normalize
            scans_cutout[pt_idx, scan_idx, :] = cutout_sampled

    Parallel(n_jobs=2)(delayed(_ct_pt)(pt_idx) for pt_idx in pt_inds)

    return scans_cutout


def scans_to_polar_grid(scans, min_range=0.0, max_range=30.0, range_bin_size=1.0,
                        tsdf_clip=1.0, normalize=True):
    num_scans, num_pts = scans.shape
    num_range = int((max_range - min_range) / range_bin_size) + 1
    mag_range, mid_range = max_range - min_range, 0.5 * (max_range - min_range)

    polar_grid = np.empty((num_scans, num_range, num_pts), dtype=np.float32)

    scans = np.clip(scans, min_range, max_range)
    scans_grid_inds = ((scans - min_range) / range_bin_size).astype(np.int32)

    for i_scan in range(num_scans):
        for i_pt in range(num_pts):
            range_grid_ind = scans_grid_inds[i_scan, i_pt]
            scan_val = scans[i_scan, i_pt]

            if tsdf_clip > 0.0:
                min_dist, max_dist = 0 - range_grid_ind, num_range - range_grid_ind
                tsdf = np.arange(min_dist, max_dist, step=1).astype(np.float32) * range_bin_size
                tsdf = np.clip(tsdf, -tsdf_clip, tsdf_clip)
            else:
                tsdf = np.zeros(num_range, dtype=np.float32)

            if normalize:
                scan_val = (scan_val - mid_range) / mag_range * 2.0
                tsdf = tsdf / mag_range * 2.0

            tsdf[range_grid_ind] = scan_val
            polar_grid[i_scan, :, i_pt] = tsdf

    return polar_grid


def group_predicted_center(scan_grid, phi_grid, pred_cls, pred_reg, min_thresh=1e-5,
                           class_weights=None, bin_size=0.1, blur_sigma=0.5,
                           x_min=-15.0, x_max=15.0, y_min=-5.0, y_max=15.0,
                           vote_collect_radius=0.3, cls_agnostic_vote=False):
    '''
    Convert a list of votes to a list of detections based on Non-Max suppression.

    ` `vote_combiner` the combination function for the votes per detection.
    - `bin_size` the bin size (in meters) used for the grid where votes are cast.
    - `blur_win` the window size (in bins) used to blur the voting grid.
    - `blur_sigma` the sigma used to compute the Gaussian in the blur window.
    - `x_min` the left limit for the voting grid, in meters.
    - `x_max` the right limit for the voting grid, in meters.
    - `y_min` the bottom limit for the voting grid in meters.
    - `y_max` the top limit for the voting grid in meters.
    - `vote_collect_radius` the radius use during the collection of votes assigned
      to each detection.

    Returns a list of tuples (x,y,probs) where `probs` has the same layout as
    `probas`.
    '''
    pred_r, pred_phi = canonical_to_global(scan_grid, phi_grid,
                                           pred_reg[:,0], pred_reg[:, 1])
    pred_xs, pred_ys = rphi_to_xy(pred_r, pred_phi)

    instance_mask = np.zeros(len(scan_grid), dtype=np.int32)
    scan_array_inds = np.arange(len(scan_grid))

    single_cls = pred_cls.shape[1] == 1

    if class_weights is not None and not single_cls:
        pred_cls = np.copy(pred_cls)
        pred_cls[:, 1:] *= class_weights

    # voting grid
    x_range = int((x_max-x_min) / bin_size)
    y_range = int((y_max-y_min) / bin_size)
    grid = np.zeros((x_range, y_range, pred_cls.shape[1]), np.float32)

    # update x/y max to correspond to the end of the last bin.
    x_max = x_min + x_range * bin_size
    y_max = y_min + y_range * bin_size

    # filter out all the weak votes
    pred_cls_agn = pred_cls[:, 0] if single_cls else np.sum(pred_cls[:, 1:], axis=-1)
    voters_inds = np.where(pred_cls_agn > min_thresh)[0]

    if len(voters_inds) == 0:
        return [], [], instance_mask

    pred_xs, pred_ys = pred_xs[voters_inds], pred_ys[voters_inds]
    pred_cls = pred_cls[voters_inds]
    scan_array_inds = scan_array_inds[voters_inds]
    pred_x_inds = np.int64((pred_xs - x_min) / bin_size)
    pred_y_inds = np.int64((pred_ys - y_min) / bin_size)

    # discard out of bound votes
    mask = (0 <= pred_x_inds) & (pred_x_inds < x_range) & (0 <= pred_y_inds) & (pred_y_inds < y_range)
    pred_x_inds, pred_xs = pred_x_inds[mask], pred_xs[mask]
    pred_y_inds, pred_ys = pred_y_inds[mask], pred_ys[mask]
    pred_cls = pred_cls[mask]
    scan_array_inds = scan_array_inds[mask]

    # vote into the grid, including the agnostic vote as sum of class-votes!
    # @TODO Do we need the class grids?
    if single_cls:
        np.add.at(grid, (pred_x_inds, pred_y_inds), pred_cls)
    else:
        np.add.at(grid, (pred_x_inds, pred_y_inds),
                  np.concatenate([np.sum(pred_cls[:, 1:], axis=1, keepdims=True),
                                 pred_cls[:, 1:]],
                                 axis=1))

    # NMS, only in the "common" voting grid
    grid_all_cls = grid[:, :, 0]
    if blur_sigma > 0:
        blur_win = int(2 * ((blur_sigma*5) // 2) + 1)
        grid_all_cls = cv2.GaussianBlur(grid_all_cls, (blur_win, blur_win), blur_sigma)
    grid_nms_val = maximum_filter(grid_all_cls, size=3)
    grid_nms_inds = (grid_all_cls == grid_nms_val) & (grid_all_cls > 0)
    nms_xs, nms_ys = np.where(grid_nms_inds)

    if len(nms_xs) == 0:
        return [], [], instance_mask

    # Back from grid-bins to real-world locations.
    nms_xs = nms_xs * bin_size + x_min + bin_size / 2
    nms_ys = nms_ys * bin_size + y_min + bin_size / 2

    # For each vote, get which maximum/detection it contributed to.
    # Shape of `distance_to_center` (ndets, voters) and outer is (voters)
    distance_to_center = np.hypot(pred_xs - nms_xs[:, None], pred_ys - nms_ys[:, None])
    detection_ids = np.argmin(distance_to_center, axis=0)

    # Generate the final detections by average over their voters.
    dets_xs, dets_ys, dets_cls = [], [], []
    for ipeak in range(len(nms_xs)):
        voter_inds = np.where(detection_ids == ipeak)[0]
        voter_inds = voter_inds[distance_to_center[ipeak, voter_inds] < vote_collect_radius]

        support_xs, support_ys = pred_xs[voter_inds], pred_ys[voter_inds]
        support_cls = pred_cls[voter_inds]

        # mark instance, 0 is the background
        instance_mask[scan_array_inds[voter_inds]] = ipeak + 1

        if cls_agnostic_vote and not single_cls:
            weights = np.sum(support_cls[:, 1:], axis=1)
            norm = 1.0 / np.sum(weights)
            dets_xs.append(norm * np.sum(weights * support_xs))
            dets_ys.append(norm * np.sum(weights * support_ys))
            dets_cls.append(norm * np.sum(weights[:, None] * support_cls, axis=0))
        else:
            dets_xs.append(np.mean(support_xs))
            dets_ys.append(np.mean(support_ys))
            dets_cls.append(np.mean(support_cls, axis=0))

    return np.array([dets_xs, dets_ys]).T, np.array(dets_cls), instance_mask
