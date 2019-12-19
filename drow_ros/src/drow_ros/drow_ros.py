import numpy as np

import rospy
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point

from frame_msgs.msg import DetectedPerson, DetectedPersons

import utils
from drow import prediction_to_detection, Detector

# import time


def people_filter(xs, ys, confs, thres=0.0):
    labels = np.argmax(confs, axis=1)
    is_people = np.logical_and(labels == 3, confs[:, 3] > thres)
    return xs[is_people], ys[is_people], confs[is_people]


class DrowRos():
    """ROS node to detect pedestrian using DROW."""

    def __init__(self):
        self._detection_id = 10000
        self._read_params()
        self._net = Detector(self.weight_file)
        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.weight_file = rospy.get_param("~weight_file")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = utils.read_publisher_param("visual_marker")
        self._visual_pub = rospy.Publisher(
                topic, MarkerArray, queue_size=queue_size, latch=latch)

        topic, queue_size, latch = utils.read_publisher_param("detections")
        self._dets_pub = rospy.Publisher(
                topic, DetectedPersons, queue_size=queue_size, latch=latch)

        # Subscriber
        topic, queue_size = utils.read_subscriber_param("scan")
        self._scan_sub = rospy.Subscriber(
                topic, LaserScan, self._scan_callback, queue_size=queue_size)

    def _scan_callback(self, msg):
        if not self._net.initialized():
            self._net.init(msg.angle_min, msg.angle_max, len(msg.ranges))

        if self._dets_pub.get_num_connections() == 0:
            return

        scan = np.array(msg.ranges)

        # Inference
        # t = time.time()
        xs_pred, ys_pred, confs_pred = self._net(scan)
        xs_det, ys_det, confs_det = prediction_to_detection(xs_pred, ys_pred, confs_pred)
        xs_det, ys_det, confs_det = people_filter(xs_det, ys_det, confs_det)
        # print(t - time.time())

        dps = DetectedPersons()
        dps.header = msg.header
        for x, y, conf in zip(xs_det, ys_det, confs_det):
            dp = DetectedPerson()
            dp.modality = DetectedPerson.MODALITY_GENERIC_LASER_2D
            dp.confidence = conf[-1]
            dp.pose.pose.position.x = y
            dp.pose.pose.position.y = -x
            dp.pose.pose.position.z = 0.7
            dp.pose.pose.orientation.w = 1.0
            # These covariance values are not used for the moment, just
            # place-holder.
            covar = np.zeros(36, dtype=np.float64)
            for i in (0, 1, 2):
                covar[i * 6 + i] = 0.05
            for i in (3, 4, 5):
                covar[i * 6 + i] = 1e6
            dp.pose.covariance = covar
            dp.bbox_x, dp.bbox_y = 0.0, 0.0
            dp.bbox_w, dp.bbox_h = 0.0, 0.0
            dp.detection_id = self._detection_id
            self._detection_id += 20  # @TODO
            dp.height = 1.85  # @TODO
            dp.warp_loss = 0.0
            dps.detections.append(dp)

        self._dets_pub.publish(dps)

        # RViz
        if self._visual_pub.get_num_connections() > 0:
            ms_msg = MarkerArray()

            # Prediction
            m_msg = Marker()
            m_msg.header = msg.header
            m_msg.action = Marker.ADD
            m_msg.type = Marker.POINTS
            xs_pred, ys_pred, confs_pred = people_filter(xs_pred, ys_pred, confs_pred, thres=0)
            m_msg.id = 0
            m_msg.scale.x, m_msg.scale.y, m_msg.scale.z = 0.1, 0.1, 0.1
            m_msg.color.r, m_msg.color.g, m_msg.color.b = 0.0, 1.0, 0.0
            m_msg.color.a = 0.5
            points = []
            for x, y in zip(xs_pred, ys_pred):
                p = Point()
                p.x, p.y, p.z = y, -x, 0.5
                points.append(p)
            m_msg.points = points
            ms_msg.markers.append(m_msg)

            # Detection
            m_msg = Marker()
            m_msg.header = msg.header
            m_msg.action = Marker.ADD
            m_msg.type = Marker.POINTS
            m_msg.id = 1
            m_msg.scale.x, m_msg.scale.y, m_msg.scale.z = 0.25, 0.25, 0.25
            m_msg.color.r, m_msg.color.g, m_msg.color.b = 1.0, 1.0, 0.0
            m_msg.color.a = 1.0
            points = []
            for x, y in zip(xs_det, ys_det):
                p = Point()
                p.x, p.y, p.z = y, -x, 0.5
                points.append(p)
            m_msg.points = points
            ms_msg.markers.append(m_msg)
            self._visual_pub.publish(ms_msg)
