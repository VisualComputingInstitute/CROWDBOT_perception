import numpy as np
import rospy

def rphi_to_xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan_to_xy(scan, thresh=104.0):
    s = scan
    if thresh is not None:
        s[s > thresh] = np.nan
    angles = np.linspace(-225.0 / 2, 225.0 / 2, len(scan)) / 180.0 * np.pi
    return rphi_to_xy(s, angles)


def xy_to_rphi(x, y):
    # NOTE: Axes rotated by 90 CCW by intent, so that 0 is top.
    return np.hypot(x, y), np.arctan2(-x, y)

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

