#!/usr/bin/env python

import rospy
from drow_ros.drow_ros import DROWRos


if __name__ == '__main__':
    rospy.init_node('drow_ros')
    try:
        DROWRos()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()