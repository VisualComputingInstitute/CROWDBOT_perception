from threading import Lock
import cv2
import numpy as np
import scipy.ndimage as ndimage

import rospy
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image

from frame_msgs.msg import DetectedPerson, DetectedPersons

import utils

def fuse_detections_simple(dets_queue, xmin, xmax, ymin, ymax, dist_thres):
    dets_fused = []
    # Use latest detections first
    for dets in reversed(dets_queue):
        for det in dets.detections:
            if det.pose.pose.position.x < xmin \
                    or det.pose.pose.position.x > xmax \
                    or det.pose.pose.position.y < ymin \
                    or det.pose.pose.position.y > ymax:
                continue

            idx_dup = find_duplicate_detections(det, dets_fused, dist_thres)

            if idx_dup is None:
                dets_fused.append(det)
            else:
                det_dup = dets_fused[idx_dup]
                if det_dup.modality != DetectedPerson.MODALITY_GENERIC_MONOCULAR_VISION \
                        and det.modality == DetectedPerson.MODALITY_GENERIC_MONOCULAR_VISION:
                    dets_fused[idx_dup] = det

    return dets_fused


def distance_between_detections(det0, det1):
    return np.hypot(det0.pose.pose.position.x - det1.pose.pose.position.x,
                    det0.pose.pose.position.y - det1.pose.pose.position.y)


def find_duplicate_detections(new_det, dets_queue, dist_thres):
    for idx, det in enumerate(dets_queue):
        dist = distance_between_detections(det, new_det)
        if dist < dist_thres:
            return idx
    return None


def fuse_detections_voting(dets_queue, xmin, xmax, ymin, ymax, bin_size, blur_size,
                           nms_size, dist_thres):
    # Get voting grids
    x_number_bins = int((xmax - xmin) / bin_size)
    y_number_bins = int((ymax - ymin) / bin_size)
    xmax = xmin + x_number_bins * bin_size
    ymax = ymin + y_number_bins * bin_size
    pad_size = 25
    votes = np.zeros((x_number_bins + 2 * pad_size, y_number_bins + 2 * pad_size),
                     dtype=np.float32)

    # For looking up detection later.
    lidar_det_xs, lidar_det_ys, lidar_det_indices = [], [], []
    cam_det_xs, cam_det_ys, cam_det_indices = [], [], []

    # Collect detections to voting grid.
    time_now = dets_queue[-1].header.stamp.secs
    for i, dets in enumerate(dets_queue):
        time_diff = abs(time_now - dets.header.stamp.secs)
        kernel = get_voting_kernel(time_diff, bin_size)
        ksize_half = (kernel.shape[0] - 1) / 2
        for j, det in enumerate(dets.detections):
            if not utils.posestamped_in_bound(det.pose, xmin, xmax, ymin, ymax):
                continue
            if det.modality == DetectedPerson.MODALITY_GENERIC_MONOCULAR_VISION:
                cam_det_xs.append(det.pose.pose.position.x)
                cam_det_ys.append(det.pose.pose.position.y)
                cam_det_indices.append((i, j))
            else:
                lidar_det_xs.append(det.pose.pose.position.x)
                lidar_det_ys.append(det.pose.pose.position.y)
                lidar_det_indices.append((i, j))
            x_idx = int((det.pose.pose.position.x - xmin) / bin_size)
            y_idx = int((det.pose.pose.position.y - ymin) / bin_size)
            x0 = x_idx - ksize_half + pad_size
            x1 = x_idx + ksize_half + pad_size + 1
            y0 = y_idx - ksize_half + pad_size
            y1 = y_idx + ksize_half + pad_size + 1
            votes[x0:x1, y0:y1] = votes[x0:x1, y0:y1] + kernel * det.confidence  # Weight by confidence
    votes = votes[pad_size:-pad_size, pad_size:-pad_size]
    has_lidar = len(lidar_det_indices) > 0

    # Find peaks in the voting grid.
    votes = cv2.GaussianBlur(votes, (blur_size, blur_size), 0)
    nms_size = 2 * int(0.5 / bin_size) + 1  # @TODO
    votes_max = ndimage.maximum_filter(votes, size=nms_size)
    is_max = np.logical_and(votes > 0.0, votes == votes_max)
    max_xs_idx, max_ys_idx = np.where(is_max)
    max_xs = max_xs_idx * bin_size + xmin + bin_size / 2.0
    max_ys = max_ys_idx * bin_size + ymin + bin_size / 2.0

    # Find the distance between detections and peaks in voting grid.
    # [i, j] is the square distance of jth detection to ith peak.
    cam_det_xs = np.array(cam_det_xs, dtype=np.float32)
    cam_det_ys = np.array(cam_det_ys, dtype=np.float32)
    cam_dist_to_max_sq = np.square(cam_det_xs - max_xs[:, np.newaxis]) \
            + np.square(cam_det_ys - max_ys[:, np.newaxis])

    # Keep track if the detection has been assigned to a peak already.
    cam_is_free = np.ones(len(cam_det_indices), dtype=np.bool)

    if has_lidar:
        lidar_det_xs = np.array(lidar_det_xs, dtype=np.float32)
        lidar_det_ys = np.array(lidar_det_ys, dtype=np.float32)
        lidar_dist_to_max_sq = np.square(lidar_det_xs - max_xs[:, np.newaxis]) \
                + np.square(lidar_det_ys - max_ys[:, np.newaxis])
        lidar_is_free = np.ones(len(lidar_det_indices), dtype=np.bool)

    # Fuse detections.
    dets_fused = []
    for peak_id, (x, y, cam_dist) \
            in enumerate(zip(max_xs, max_ys, cam_dist_to_max_sq)):
        # Find the nearest detection from camera. If the nearest detection falls
        # within threshold, assign it to peak.
        det = None
        cam_min, cam_min_idx = find_min_true_val(cam_dist,
                                                 cam_is_free)
        if np.sqrt(cam_min) < dist_thres:
            cam_is_free[cam_min_idx] = False
            det_idx_i, det_idx_j = cam_det_indices[cam_min_idx]
            det = dets_queue[det_idx_i].detections[det_idx_j]
        elif has_lidar:
        # If there is no camera detection near the peak, find if there is
        # near-by lidar detection.
            lidar_dist = lidar_dist_to_max_sq[peak_id]
            lidar_min, lidar_min_idx = find_min_true_val(lidar_dist,
                                                         lidar_is_free)
            if np.sqrt(lidar_min) < dist_thres:
                lidar_is_free[lidar_min_idx] = False
                det_idx_i, det_idx_j = lidar_det_indices[lidar_min_idx]
                det = dets_queue[det_idx_i].detections[det_idx_j]

        # Store the detection to output detection array, adjusting x and y
        if det is not None:
            det.pose.pose.position.x = x
            det.pose.pose.position.y = y
            dets_fused.append(det)

    return dets_fused, votes


def find_min_true_val(vals, is_true):
    """
    @brief      Given an array of value and an array of binary mask, find the
                smallest value whose binary flag is true.
    """
    indices = vals.argsort()
    vals = vals[indices]
    is_true = is_true[indices]
    i, imax = 0, len(indices)
    while i < imax and not is_true[i]:
        i += 1

    if i < imax:
        return vals[i], indices[i]
    else:
        return np.nan, np.nan


def get_voting_kernel(time_diff, bin_size):
    vel = 3.0  # m/s
    ksize = 2 * int(vel * time_diff / bin_size) + 1
    # @TODO Circular kernel
    k = np.ones((ksize, ksize), dtype=np.float32)
    return k / k.size + 1e-6 * np.random.rand(ksize, ksize)  # Break ties

class DetectionFusion():
    """
    @brief      This ROS node listens to asynchronous 3D detections from
                multiple sources and fuse them into a synchronous detection
                output. Duplicate detections are merged together based on
                spacial proximity.
    """

    def __init__(self):
        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._lock = Lock()
        self._dets_msg_queue = []

        self._cam_det_id = 5   # @TODO
        self._lidar_det_id = 1000

        self._read_params()

        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.number_of_detections = rospy.get_param("~number_of_detections")
        self.time_span_between_detections = rospy.get_param("~time_span_between_detections")
        self.detection_sources = rospy.get_param("~number_of_detection_source")
        self.fixed_frame = rospy.get_param("~fixed_frame")
        self.robot_frame = rospy.get_param("~robot_frame")

        self.xmin = rospy.get_param("~voting_grid_x_min")
        self.xmax = rospy.get_param("~voting_grid_x_max")
        self.ymin = rospy.get_param("~voting_grid_y_min")
        self.ymax = rospy.get_param("~voting_grid_y_max")
        self.bin_size = rospy.get_param("~voting_grid_bin_size")

        self.blur_size = rospy.get_param("~voting_blur_size")
        self.nms_size = rospy.get_param("~voting_nms_size")
        self.dist_thres = rospy.get_param("~voting_dist_threshold")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = utils.read_publisher_param("detections")
        self._detections_pub = rospy.Publisher(
                topic, DetectedPersons, queue_size=queue_size, latch=latch)

        topic, queue_size, latch = utils.read_publisher_param("detection_votes_image")
        self._votes_pub = rospy.Publisher(
                topic, Image, queue_size=queue_size, latch=latch)

        # Subscriber
        self._dets_sub = []
        for i in range(self.detection_sources):
            topic, queue_size = utils.read_subscriber_param("detections" + str(i))
            self._dets_sub.append(rospy.Subscriber(
                                        topic,
                                        DetectedPersons,
                                        self._detections_callback,
                                        queue_size=queue_size))

    def _should_fuse(self):
        time_diff = float(self._dets_msg_queue[-1].header.stamp.secs \
                          - self._dets_msg_queue[0].header.stamp.secs)
        return len(self._dets_msg_queue) > self.number_of_detections \
                or time_diff > self.time_span_between_detections

    def _transform_detected_persons_msg(self, msg, target_frame, timeout=2.0):
        try:
            trans = self._tf_buffer.lookup_transform(target_frame,
                                                     msg.header.frame_id,
                                                     msg.header.stamp,
                                                     rospy.Duration(timeout))
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException):
            print("[DetectionFusion] Could not find transformation. ",
                    "Target frame: {}, source frame: {}, timeout: {}".format(
                    target_frame, msg.header.frame_id, timeout))
            return

        msg.header.frame_id = target_frame
        for i, dp in enumerate(msg.detections):
            posestamped_target_frame = \
                    tf2_geometry_msgs.do_transform_pose(dp.pose, trans)
            dp.pose.pose = posestamped_target_frame.pose

        return msg

    def _detections_callback(self, msg):
        if self._detections_pub.get_num_connections() == 0:
            return

        msg = self._transform_detected_persons_msg(msg, self.robot_frame)
        if msg is None:
            return

        # Add detections to queue, and check if we should fuse detection.
        with self._lock:
            self._dets_msg_queue.append(msg)
            if self._should_fuse():
                dets_msg_queue, self._dets_msg_queue = self._dets_msg_queue, []
            else:
                return

        # Fuse detections
        # dets_fused, votes = fuse_detections_voting(
        #         dets_msg_queue, self.xmin, self.xmax, self.ymin, self.ymax,
        #         self.bin_size, self.blur_size, self.nms_size, self.dist_thres)

        dets_fused = fuse_detections_simple(
                dets_msg_queue, self.xmin, self.xmax, self.ymin, self.ymax,
                self.dist_thres)
        votes = None

        # Publish in fixed frame
        dps = DetectedPersons()
        dps.header = dets_msg_queue[-1].header
        dps.detections = dets_fused
        dps = self._transform_detected_persons_msg(dps, self.fixed_frame)
        if dps is None:
            return
        # @TODO
        for dp in dps.detections:
            if dp.modality == DetectedPerson.MODALITY_GENERIC_MONOCULAR_VISION:
                dp.detection_id = self._cam_det_id
                self._cam_det_id += 20  # Use 20 will have same color in RViz visualization
            else:
                dp.detection_id = self._lidar_det_id
                self._lidar_det_id += 20  # Use 20 will have same color in RViz visualization
        self._detections_pub.publish(dps)

        if votes is not None and self._votes_pub.get_num_connections() > 0:
            votes = votes / (np.max(votes) + 0.0001)
            try:
                self._votes_pub.publish(
                        self._bridge.cv2_to_imgmsg(votes, "32FC1"))
            except CvBridgeError as e:
                print("[DetectionFusion]", e)

