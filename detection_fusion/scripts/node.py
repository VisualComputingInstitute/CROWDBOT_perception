#!/usr/bin/env python

import rospy
from detection_fusion.detection_fusion import DetectionFusion


if __name__ == '__main__':
    rospy.init_node('detection_fusion')
    try:
        DetectionFusion()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()