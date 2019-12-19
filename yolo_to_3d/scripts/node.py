#!/usr/bin/env python

import rospy
from yolo_to_3d.yolo_to_3d import YoloTo3D


if __name__ == '__main__':
    rospy.init_node('yolo_to_3d')
    try:
        YoloTo3D()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()