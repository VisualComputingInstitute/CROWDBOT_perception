#!/usr/bin/env python
import glob
import os

import numpy as np

import rospy
import rosbag

from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import LaserScan
from tf2_msgs.msg import TFMessage


def load_data(fname):
    data = np.genfromtxt(fname, delimiter=",")
    seqs, times, scans = data[:,0].astype(np.uint32), data[:,1].astype(np.float32), data[:,2:].astype(np.float32)
    return seqs, times, scans


def sequence_to_bag(seq_name, bag_name):
    scan_msg = LaserScan()
    scan_msg.header.frame_id = 'sick_laser_front'
    scan_msg.angle_min = np.radians(-225.0 / 2)
    scan_msg.angle_max = np.radians(225.0 / 2)
    scan_msg.range_min = 0.005
    scan_msg.range_max = 100.0
    scan_msg.scan_time = 0.066667
    scan_msg.time_increment = 0.000062
    scan_msg.angle_increment = (scan_msg.angle_max - scan_msg.angle_min) / 450

    tran = TransformStamped()
    tran.header.frame_id = 'base_footprint'
    tran.child_frame_id = 'sick_laser_front'
    tran.transform.translation.x = 0.0
    tran.transform.translation.y = 0.0
    tran.transform.translation.z = 0.0
    tran.transform.rotation.x = 0.0
    tran.transform.rotation.y = 0.0
    tran.transform.rotation.z = 0.0
    tran.transform.rotation.w = 1.0
    tf_msg = TFMessage([tran])

    with rosbag.Bag(bag_name, 'w') as bag:
        seqs, times, scans = load_data(seq_name)
        for seq, time, scan in zip(seqs, times, scans):
            time = rospy.Time(time)
            scan_msg.header.seq = seq
            scan_msg.header.stamp = time
            scan_msg.ranges = scan
            bag.write('/sick_laser_front/scan', scan_msg, t=time)
            tran.header.stamp = time
            bag.write('/tf', tf_msg, t=time)


def convert_all():
    data_dir = "/fastwork/jia/data/DROWv2-data/"
    train_names = [f[:-4] for f in glob.glob(os.path.join(data_dir, 'train', '*.csv'))]
    val_names = [f[:-4] for f in glob.glob(os.path.join(data_dir, 'val', '*.csv'))]

    output_dir = "/work/jia/data/DROW-bag"
    train_dir = os.path.join(output_dir, 'train')
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    val_dir = os.path.join(output_dir, 'val')
    if not os.path.exists(val_dir):
        os.makedirs(val_dir)

    for i, f_name in enumerate(train_names):
        print('Convert training data: seq {}/{}'.format(i+1, len(train_names)))
        bag_name = os.path.join(train_dir, os.path.split(f_name)[-1])
        seq_name = f_name + '.csv'
        sequence_to_bag(seq_name, bag_name)

    for i, f_name in enumerate(val_names):
        print('Convert val data: seq {}/{}'.format(i+1, len(val_names)))
        bag_name = os.path.join(val_dir, os.path.split(f_name)[-1])
        seq_name = f_name + '.csv'
        sequence_to_bag(seq_name, bag_name)


if __name__ == '__main__':
    convert_all()