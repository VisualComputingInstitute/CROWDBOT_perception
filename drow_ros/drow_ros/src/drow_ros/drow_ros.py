import numpy as np
import rospy
from sensor_msgs.msg import LaserScan
from frame_msgs.msg import DetectedPerson, DetectedPersons
from drow.drow_detector import DROWDetector
# import time

def read_subscriber_param(name):
    """
    @brief      Convenience function to read subscriber parameter.
    """
    topic = rospy.get_param("~subscriber/" + name + "/topic")
    queue_size = rospy.get_param("~subscriber/" + name + "/queue_size")
    return topic, queue_size


def read_publisher_param(name):
    """
    @brief      Convenience function to read publisher parameter.
    """
    topic = rospy.get_param("~publisher/" + name + "/topic")
    queue_size = rospy.get_param("~publisher/" + name + "/queue_size")
    latch = rospy.get_param("~publisher/" + name + "/latch")
    return topic, queue_size, latch


class DROWRos():
    """ROS node to detect pedestrian using DROW."""
    def __init__(self):
        self._detection_id = 0
        self._read_params()
        self._drow = DROWDetector(self.weight_file)
        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.weight_file = rospy.get_param("~weight_file")
        self.conf_thresh = rospy.get_param("~conf_thresh")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = read_publisher_param("detections")
        self._dets_pub = rospy.Publisher(
                topic, DetectedPersons, queue_size=queue_size, latch=latch)

        # Subscriber
        topic, queue_size = read_subscriber_param("scan")
        self._scan_sub = rospy.Subscriber(
                topic, LaserScan, self._scan_callback, queue_size=queue_size)

    def _scan_callback(self, msg):
        if self._dets_pub.get_num_connections() == 0:
            return

        if not self._drow.laser_spec_set():
            self._drow.set_laser_spec(angle_inc=msg.angle_increment,
                                      num_pts=len(msg.ranges))

        # t = time.time()
        scan = np.array(msg.ranges)
        dets_xs, dets_ys, dets_cls = self._drow(scan)
        # print(t - time.time())

        # only keep pedestrian
        dets_cls_label = np.argmax(dets_cls[:, 1:], axis=1) + 1
        dets_cls_conf = np.max(dets_cls[:, 1:], axis=1)
        pedestrian_mask = dets_cls_label == 3  # 1 - wheelchair, 2 - people with walking aids, 3 - people
        dets_xs = dets_xs[pedestrian_mask]
        dets_ys = dets_ys[pedestrian_mask]
        dets_cls_conf = dets_cls_conf[pedestrian_mask]

        # confidence threshold
        conf_mask = dets_cls_conf >= self.conf_thresh
        dets_xs = dets_xs[conf_mask]
        dets_ys = dets_ys[conf_mask]
        dets_cls_conf = dets_cls_conf[conf_mask]

        # convert and publish ros msg
        dps_msg = self._detections_to_ros_msg(dets_xs, dets_ys, dets_cls_conf)
        dps_msg.header = msg.header
        self._dets_pub.publish(dps_msg)


    def _detections_to_ros_msg(self, dets_xs, dets_ys, dets_conf):
        dps = DetectedPersons()
        for x, y, conf in zip(dets_xs, dets_ys, dets_conf):
            dp = DetectedPerson()
            dp.modality = DetectedPerson.MODALITY_GENERIC_LASER_2D
            dp.confidence = conf
            # If laser is facing front, DROW's y-axis aligns with the laser
            # center ray, x-axis points to right
            dp.pose.pose.position.x = y
            dp.pose.pose.position.y = -x
            dp.pose.pose.position.z = 1.0
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
            dp.height = 1.85
            dp.warp_loss = 0.5  # 0.5 is neutral
            dp.detection_id = self._detection_id
            dps.detections.append(dp)
            self._detection_id += 1