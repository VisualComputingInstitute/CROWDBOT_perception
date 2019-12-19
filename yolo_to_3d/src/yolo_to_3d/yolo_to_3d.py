from threading import Lock
import cv2
import numpy as np

import rospy
import tf2_ros
import tf2_geometry_msgs

from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker

from frame_msgs.msg import DetectedPerson, DetectedPersons
from rwth_perception_people_msgs.msg import GroundPlane
from darknet_ros_msgs.msg import BoundingBoxes

import utils


def boxes_to_3d_with_ground_plane(boxes, K_inv, gp_n, gp_d):
    nb = boxes.shape[1]

    # Choose bottom center of bbox as starting point of ray casting.
    points = np.empty((3, nb), dtype=np.float32)
    points[0, :] = (boxes[0, :] + boxes[2, :]) / 2
    points[1, :] = boxes[3, :]
    points[2, :] = 1
    points = np.matmul(K_inv, points)

    # Compute intercepting points with ground plane in camera frame.
    scale = gp_d / np.matmul(gp_n, points)
    points = scale * points
    s_mask = points[2, :] > 0  # Success mask

    return points, s_mask


def boxes_to_3d_with_depth(boxes, K_inv, depth,
                           range_min, range_max, min_valid_ratio):
    nb = boxes.shape[1]
    s_mask = np.ones(nb, dtype=bool)  # Success mask

    # Choose center pixel of box as starting point of ray casting.
    points = np.empty((3, nb), dtype=np.float32)
    points[0, :] = (boxes[0, :] + boxes[2, :]) / 2
    points[1, :] = (boxes[1, :] + boxes[3, :]) / 2
    points[2, :] = 1
    points = np.matmul(K_inv, points)

    for i, box in enumerate(boxes.transpose()):
        width = box[2] - box[0]
        height = box[3] - box[1]
        if width < 10 or height < 10:
            s_mask[i] = False
            continue

        # Extract depth measurement at the bbox central region.
        w_quat = int(0.25 * width)
        h_quat = int(0.25 * height)
        x0 = box[0] + w_quat
        x1 = box[2] - w_quat + 1
        y0 = box[1] + h_quat
        y1 = box[3] - h_quat + 1
        depth_roi = depth[y0:y1, x0:x1]

        # Check if most of the depth measurement is not NaN of Inf.
        is_valid = np.isfinite(depth_roi)
        valid_ratio = np.sum(is_valid) / depth_roi.size
        if valid_ratio < min_valid_ratio:
            s_mask[i] = False
            continue

        # Use median value of depth within central region.
        d_med = np.median(depth_roi)
        if d_med < range_min or d_med > range_max:
            s_mask[i] = False
            continue

        points[0, i] = points[0, i] / points[2, i] * d_med
        points[1, i] = points[1, i] / points[2, i] * d_med
        points[2, i] = d_med

    return points, s_mask


def yolo_boxes_to_numpy(msg):
    boxes = np.empty((4, len(msg.bounding_boxes)), dtype=int)
    for i, box_msg in enumerate(msg.bounding_boxes):
        boxes[0, i] = box_msg.xmin
        boxes[1, i] = box_msg.ymin
        boxes[2, i] = box_msg.xmax
        boxes[3, i] = box_msg.ymax

    return boxes


class YoloTo3D():
    """
    @brief      This ROS node listens to 2D detections from yolo, and converts
                them to 3D, through intercepting rays with ground plane or depth
                measurement.
    """

    def __init__(self):
        self._read_params()

        self._bridge = CvBridge()
        self._tf_buffer = tf2_ros.Buffer(rospy.Duration(10))
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)

        self._cam_frame = None
        self._cam_K_inv = None
        self._prev_yolo_msg_seq = None

        if self.use_measured_depth:
            self._depth_lock = Lock()
            self._depth = None
            self._depth_time = None
        else:
            self._gp_lock = Lock()
            self._gp_frame = None
            self._gp_n = None
            self._gp_d = None

        if self.publish_bounding_box_image:
            self._image_lock = Lock()
            self._image = None

        self._det_id = 0   # @TODO

        self._init()

    def _read_params(self):
        """
        @brief      Reads parameters from ROS server.
        """
        self.use_measured_depth = rospy.get_param("~use_measured_depth")
        self.depth_time_threshold = rospy.get_param("~depth_time_threshold")
        self.depth_min_valid_ratio = rospy.get_param("~depth_min_valid_ratio")
        self.depth_scale = rospy.get_param("~depth_scale")
        self.depth_min_median = rospy.get_param("~depth_min_median")
        self.depth_max_median = rospy.get_param("~depth_max_median")
        self.publish_bounding_box_image = rospy.get_param("~publish_bounding_box_image")

    def _init(self):
        """
        @brief      Initialize ROS connection.
        """
        # Publisher
        topic, queue_size, latch = utils.read_publisher_param("detected_persons")
        self._detections_pub = rospy.Publisher(
                topic, DetectedPersons, queue_size=queue_size, latch=latch)

        topic, queue_size, latch = utils.read_publisher_param("visual_marker")
        self._visual_pub = rospy.Publisher(
                topic, Marker, queue_size=queue_size, latch=latch)

        # Subscriber
        topic, queue_size = utils.read_subscriber_param("camera_info")
        self._camera_info_sub = rospy.Subscriber(
                topic, CameraInfo, self._camera_info_callback, queue_size=queue_size)

        topic, queue_size = utils.read_subscriber_param("yolo_bounding_boxes")
        self._yolo_bounding_boxes_sub = rospy.Subscriber(
                topic, BoundingBoxes, self._yolo_bounding_boxes_callback, queue_size=queue_size)

        if self.use_measured_depth:
            topic, queue_size = utils.read_subscriber_param("depth")
            self._depth_sub = rospy.Subscriber(
                    topic, Image, self._depth_callback, queue_size=queue_size)
        else:
            topic, queue_size = utils.read_subscriber_param("ground_plane")
            self._ground_plane_sub = rospy.Subscriber(
                    topic, GroundPlane, self._ground_plane_callback, queue_size=queue_size)

        # Option to publish image with detection
        if self.publish_bounding_box_image:
            topic, queue_size, latch = utils.read_publisher_param("image")
            self._image_pub = rospy.Publisher(
                    topic, Image, queue_size=queue_size, latch=latch)

            topic, queue_size = utils.read_subscriber_param("image")
            self._image_sub = rospy.Subscriber(
                    topic, Image, self._image_callback, queue_size=queue_size)

    def _camera_info_callback(self, msg):
        if self._cam_frame is not None and self._cam_K_inv is not None:
            return
        self._cam_frame = msg.header.frame_id
        self._cam_K_inv = np.linalg.inv(
                np.array(msg.K, dtype=np.float64).reshape(3, 3))

    def _ground_plane_callback(self, msg):
        with self._gp_lock:
            self._gp_frame = msg.header.frame_id.strip("/")
            self._gp_n = np.array(msg.n, dtype=np.float64)
            self._gp_d = msg.d

    def _depth_callback(self, msg):
        with self._depth_lock:
            try:
                self._depth = self._bridge.imgmsg_to_cv2(
                        msg, desired_encoding="16UC1").astype(np.float64)
                self._depth *= self.depth_scale
                self._depth_time = msg.header.stamp
            except CvBridgeError as e:
                print("[YoloTo3D]", e)

    def _image_callback(self, msg):
        with self._image_lock:
            try:
                self._image = self._bridge.imgmsg_to_cv2(
                        msg, desired_encoding="bgr8")
            except CvBridgeError as e:
                print("[YoloTo3D]", e)

    def _yolo_bounding_boxes_callback(self, msg):
        """
        @brief      Main callback.
        """
        if msg.image_header.seq == self._prev_yolo_msg_seq:
            return
        else:
            self._prev_yolo_msg_seq = msg.image_header.seq

        if self._detections_pub.get_num_connections() == 0:
            return

        if self._cam_frame is None or self._cam_K_inv is None:
            print("[YoloTo3D] No camera info.")
            return

        # Main processing
        dps, image = self._yolo_to_3d(msg)
        if dps is None:
            return

        # Publish
        self._detections_pub.publish(dps)
        if image is not None:
            try:
                self._image_pub.publish(
                        self._bridge.cv2_to_imgmsg(image, "bgr8"))
            except CvBridgeError as e:
                print("[YoloTo3D]", e)

    def _get_depth(self, time):
        with self._depth_lock:
            if self._depth is None or self._depth_time is None:
                return

            t_diff = abs((time - self._depth_time).secs)
            if t_diff > self.depth_time_threshold:
                return

            depth = self._depth.copy()
        return depth

    def _get_ground_plane(self, time, timeout=1.0):
        with self._gp_lock:
            if self._gp_n is None or self._gp_d is None or self._gp_frame is None:
                return None, None
            gp_n, gp_d, gp_frame = self._gp_n, self._gp_d, self._gp_frame

        if self._cam_frame != gp_frame:
            vec = Vector3Stamped()
            vec.vector.x, vec.vector.y, vec.vector.z = gp_n

            try:
                trans = self._tf_buffer.lookup_transform(
                        self._cam_frame, gp_frame, time, rospy.Duration(timeout))
            except (tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException) as e:
                print("[YoloTo3D]", e)
                return None, None

            # @TODO Get ride of do_transform_vector3.
            vec = tf2_geometry_msgs.do_transform_vector3(vec, trans)
            gp_n = np.array([vec.vector.x, vec.vector.y, vec.vector.z],
                            dtype=np.float32)
            gp_d = gp_d + (trans.transform.translation.x * gp_n[0] +
                           trans.transform.translation.y * gp_n[1] +
                           trans.transform.translation.z * gp_n[2])

        if self._visual_pub.get_num_connections() > 0:
            msg = Marker()
            msg.type = Marker.CUBE
            msg.header.frame_id = self._cam_frame
            x, y, z = 0.0, gp_d / gp_n[1], 0.0
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = x, y, z
            ref_vec = np.array([0, 0, 1], dtype=np.float64)
            theta = np.arccos(np.dot(ref_vec, gp_n))
            rot_axis = np.cross(ref_vec, gp_n) / np.sin(theta)
            sin_half_theta = np.sin(theta / 2.0)
            msg.pose.orientation.x = rot_axis[0] * sin_half_theta
            msg.pose.orientation.y = rot_axis[1] * sin_half_theta
            msg.pose.orientation.z = rot_axis[2] * sin_half_theta
            msg.pose.orientation.w = np.cos(theta / 2.0)
            msg.scale.x, msg.scale.y, msg.scale.z = 50.0, 50.0, 0.00001
            msg.color.r, msg.color.g, msg.color.b = 0.0, 0.0, 1.0
            msg.color.a = 0.5
            self._visual_pub.publish(msg)

        return gp_n, gp_d

    def _yolo_to_3d(self, msg):
        if len(msg.bounding_boxes) == 0:
            dps = DetectedPersons()
            dps.header = msg.image_header
            dps.detections = []
            image = None
            if self.publish_bounding_box_image \
                    and self._image_pub.get_num_connections() > 0 \
                    and self._image is not None:
                with self._image_lock:
                    image = self._image.copy()
            return dps, image

        boxes = yolo_boxes_to_numpy(msg)

        if self.use_measured_depth:
            depth = self._get_depth(msg.image_header.stamp)
            if depth is None:
                print("[YoloTo3D] No depth measurement within {} s.".format(
                        self.depth_time_threshold))
                return None, None

            points, s_mask = boxes_to_3d_with_depth(
                    boxes, self._cam_K_inv, depth,
                    range_min=self.depth_min_median,
                    range_max=self.depth_max_median,
                    min_valid_ratio=self.depth_min_valid_ratio)
        else:
            gp_n, gp_d = self._get_ground_plane(msg.image_header.stamp)
            if gp_n is None or gp_d is None:
                print("[YoloTo3D] No ground plane.")
                return None, None

            points, s_mask = boxes_to_3d_with_ground_plane(
                    boxes, self._cam_K_inv, gp_n, gp_d)

        image = None
        if self.publish_bounding_box_image \
                and self._image_pub.get_num_connections() > 0 \
                and self._image is not None:
            with self._image_lock:
                image = self._image.copy()

        dps = DetectedPersons()
        dps.header = msg.image_header
        for bbox, det, s in zip(msg.bounding_boxes, points.transpose(), s_mask):
            if not s and image is not None:
                p0 = (bbox.xmin, bbox.ymin)
                p1 = (bbox.xmax, bbox.ymax)
                color = (0, 0, 255)
                cv2.rectangle(image, p0, p1, color)
                continue

            dp = DetectedPerson()
            dp.modality = DetectedPerson.MODALITY_GENERIC_MONOCULAR_VISION
            dp.confidence = bbox.probability
            dp.pose.pose.position.x = det[0]
            dp.pose.pose.position.y = det[1]
            dp.pose.pose.position.z = det[2]
            dp.pose.pose.orientation.w = 1.0
            # These covariance values are not used for the moment, just
            # place-holder.
            covar = np.zeros(36, dtype=np.float64)
            for i in (0, 1, 2):
                covar[i * 6 + i] = 0.05
            for i in (3, 4, 5):
                covar[i * 6 + i] = 1e6
            dp.pose.covariance = covar
            dp.detection_id = self._det_id

            dp.bbox_x, dp.bbox_y = bbox.xmin, bbox.ymin
            dp.bbox_w, dp.bbox_h = bbox.xmax - bbox.xmin, bbox.ymax - bbox.ymin
            dp.height = 1.85  # @TODO
            dps.detections.append(dp)
            if image is not None:
                p0 = (bbox.xmin, bbox.ymin)
                p1 = (bbox.xmax, bbox.ymax)
                color = (0, 255, 0)
                cv2.rectangle(image, p0, p1, color)
                cv2.putText(image, str(self._det_id),
                            (bbox.xmin, bbox.ymin+10),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=color)
            self._det_id += 1

        return dps, image

