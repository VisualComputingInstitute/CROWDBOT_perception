#!/usr/bin/env python

import rospy
from drow_ros.drow_ros import DrowRos


if __name__ == '__main__':
    rospy.init_node('drow_ros')
    try:
        DrowRos()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()