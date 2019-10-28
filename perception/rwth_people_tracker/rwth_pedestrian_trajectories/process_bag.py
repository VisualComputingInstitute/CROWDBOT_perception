import rospy
from nav_msgs.msg import OccupancyGrid


if __name__ == '__main__':
    rospy.init_node('map_relay')
    pub = rospy.Publisher('/map_latched', OccupancyGrid, queue_size=1, latch=True)

    def callback(data):
        pub.publish(data)
        print(data.header)
        print(data.info)

    rospy.Subscriber('/maps/GlobalMap', OccupancyGrid, callback)
    rospy.spin()