#!/usr/bin/env python
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

import rospy
from sensor_msgs.msg import LaserScan

def load_data(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times, scans = data[:,0].astype(np.uint32), data[:,1].astype(np.float32), data[:,2:].astype(np.float32)
    return seqs, times, scans


def rphi_to_xy(r, phi):
    return r * -np.sin(phi), r * np.cos(phi)


def scan_to_xy(scan):
    s = scan
    angles = np.linspace(-225.0 / 2, 225.0 / 2, len(scan)) / 180.0 * np.pi
    return rphi_to_xy(s, angles)


def publish_data():
    scan_pub = rospy.Publisher(
            '/sick_laser_front/scan',
            LaserScan,
            queue_size=1,
            latch=False)

    msg = LaserScan()
    msg.header.seq = 0
    msg.header.frame_id = 'sick_laser_front'
    msg.angle_min = np.radians(-225.0 / 2)
    msg.angle_max = np.radians(225.0 / 2)
    msg.range_min = 0.005
    msg.range_max = 100.0
    msg.scan_time = 0.066667
    msg.time_increment = 0.000062
    msg.angle_increment = (msg.angle_max - msg.angle_min) / 450

    data_dir = "/fastwork/jia/data/DROWv2-data/"
    train_names = [f[:-4] for f in glob.glob(os.path.join(data_dir, 'train', '*.csv'))]
    # val_names = [f[:-4] for f in glob.glob(os.path.join(data_dir, 'val', '*.csv'))]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    r = rospy.Rate(10)
    for f_name in train_names:
        f_name += '.csv'
        _, _, scans = load_data(f_name)
        for scan in scans:
            if rospy.is_shutdown():
                return
            msg.header.stamp = rospy.Time.now()
            msg.ranges = scan
            # msg.intensities = np.ones_like(scan) * 10000.0
            scan_pub.publish(msg)
            msg.header.seq += 1

            scan_x, scan_y = rphi_to_xy(scan,
                    np.linspace(msg.angle_min, msg.angle_max, len(scan)))
            plt.cla()
            ax.set_xlim(-50, 50)
            ax.set_ylim(-50, 50)
            ax.scatter(scan_x, scan_y, s=1)
            plt.pause(0.001)

            r.sleep()


if __name__ == '__main__':
    rospy.init_node('drow_data_publisher')
    try:
        publish_data()
    except rospy.ROSInterruptException:
        pass
    rospy.spin()