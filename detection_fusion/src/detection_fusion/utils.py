import rospy

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

def point_in_bound(point, xmin, xmax, ymin, ymax):
    return point.x > xmin and point.x < xmax and point.y > ymin and point.y < ymax

def pose_in_bound(pose, xmin, xmax, ymin, ymax):
    return point_in_bound(pose.position, xmin, xmax, ymin, ymax)

def posestamped_in_bound(posestamped, xmin, xmax, ymin, ymax):
    return pose_in_bound(posestamped.pose, xmin, xmax, ymin, ymax)