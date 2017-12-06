// ROS includes.
#include "ros/ros.h"
#include "ros/time.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <image_transport/subscriber_filter.h>

#include <string.h>

#include "spencer_tracking_msgs/TrackedPersons.h"
#include "spencer_tracking_msgs/TrackedPersons2d.h"

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <tf/transform_datatypes.h>


using namespace std;
using namespace sensor_msgs;
using namespace message_filters;

ros::Publisher pub_tracked_persons2analyze;
ros::Publisher pub_tracked_persons2analyze_2d;

int cnt = 0;

tf::TransformListener* listener=0;
//...

void callback(const spencer_tracking_msgs::TrackedPersons::ConstPtr &trackedPersons,
              const spencer_tracking_msgs::TrackedPersons2d::ConstPtr &trackedPersons_2d)
{

    //init
    spencer_tracking_msgs::TrackedPersons filtered_tracks;
    spencer_tracking_msgs::TrackedPersons2d filtered_tracks_2d;
    filtered_tracks.header = trackedPersons->header;
    filtered_tracks_2d.header = trackedPersons_2d->header;
    spencer_tracking_msgs::TrackedPerson one_filtered_trackedPerson;
    spencer_tracking_msgs::TrackedPerson2d one_filtered_trackedPerson2d;

    //loop over all tracks
    for (int i=0; i < trackedPersons->tracks.size(); i++){
        one_filtered_trackedPerson = trackedPersons->tracks[i];
        one_filtered_trackedPerson2d = trackedPersons_2d->boxes[i];
        if(one_filtered_trackedPerson.track_id != one_filtered_trackedPerson2d.track_id){
            ROS_ERROR("Something is wrong with the IDs:\n---\nid of 3d: %d,\nid of 2d: %d\n---\n",
                      one_filtered_trackedPerson.track_id,one_filtered_trackedPerson2d.track_id);
        }

        //print orientation
        //printf("ID %d:\n",one_filtered_trackedPerson.track_id);
        //printf("v\nx:%.2f\ny:%.2f\nz:%.2f\nw:%.2f\n^\n",one_filtered_trackedPerson.pose.pose.orientation.x,
        //       one_filtered_trackedPerson.pose.pose.orientation.y,
        //       one_filtered_trackedPerson.pose.pose.orientation.z,
        //       one_filtered_trackedPerson.pose.pose.orientation.w);

        //transform orientation to camera coordinates
        string target_frame = trackedPersons_2d->header.frame_id;
        string source_frame = trackedPersons->header.frame_id;
        geometry_msgs::QuaternionStamped tempQuatStampedWorld;
        geometry_msgs::QuaternionStamped tempQuatStampedCamera;
        tempQuatStampedWorld.quaternion = one_filtered_trackedPerson.pose.pose.orientation;
        tempQuatStampedWorld.header = trackedPersons->header;
        tempQuatStampedCamera.header = trackedPersons_2d->header;
        try {
            ROS_DEBUG("Transforming received position into %s coordinate system.", target_frame.c_str());
            listener->waitForTransform(source_frame, target_frame, ros::Time(), ros::Duration(0.05));
            listener->transformQuaternion(target_frame,tempQuatStampedWorld,tempQuatStampedCamera);
        }
        catch(tf::TransformException ex) {
            ROS_WARN("Failed transform: %s", ex.what());
            return;
        }

        //orientation in camera coordinates as quaternion -> transfer to angle
        tf::Quaternion q(tempQuatStampedCamera.quaternion.x, tempQuatStampedCamera.quaternion.y,
                         tempQuatStampedCamera.quaternion.z, tempQuatStampedCamera.quaternion.w);
        tf::Matrix3x3 rot_matrix(q);
        double roll, pitch, yaw;
        rot_matrix.getRPY(roll, pitch, yaw);
        //Here (strangely?), Pitch is the important rotation!
        //std::cout << "Roll: " << roll << ", Pitch: " << pitch << ", Yaw: " << yaw << std::endl;

        // calculate the speed
        float twist_x = one_filtered_trackedPerson.twist.twist.linear.x;
        float twist_y = one_filtered_trackedPerson.twist.twist.linear.y;
        float twist_z = one_filtered_trackedPerson.twist.twist.linear.z;
        float speed = sqrt(twist_x*twist_x+twist_y*twist_y+twist_z*twist_z);
        //printf("speed is: %f\n", speed);

        //CHECK 1: angel a, keep only people looking away from camera (a=-pi/2 +|- x) and towards (a=pi/2 +|- x)
        if( (pitch<-1.3) || (pitch>1.3)){
            //CHECK 2: only keep people above certain speed x
            if (speed > 0.3){
                //...CHECK...

                //save filtered tracks
                filtered_tracks.tracks.push_back(one_filtered_trackedPerson);
                filtered_tracks_2d.boxes.push_back(one_filtered_trackedPerson2d);
            }
        }

    }

    //publish filtered tracks (3d as well as 2d)
    pub_tracked_persons2analyze.publish(filtered_tracks);
    pub_tracked_persons2analyze_2d.publish(filtered_tracks_2d);

    cnt++;

}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<spencer_tracking_msgs::TrackedPersons> &sub_tracked_persons,
                     message_filters::Subscriber<spencer_tracking_msgs::TrackedPersons2d> &sub_tracked_persons_2d){
    if(!pub_tracked_persons2analyze.getNumSubscribers()
    && !pub_tracked_persons2analyze_2d.getNumSubscribers()
    ) {
        ROS_DEBUG("Tracker2Analyze: No subscribers. Unsubscribing.");
        sub_tracked_persons.unsubscribe();
        sub_tracked_persons_2d.unsubscribe();
    } else {
        ROS_DEBUG("Tracker2Analyze: New subscribers. Subscribing.");
        sub_tracked_persons.subscribe();
        sub_tracked_persons_2d.subscribe();
    }
}


int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "analysis_planner");
    ros::NodeHandle n;

    listener = new tf::TransformListener();

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string topic_tracked_persons;
    string topic_tracked_persons_2d;

    string pub_topic_tracked_persons2analyze;
    string pub_topic_tracked_persons2analyze_2d;

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("tracked_persons", topic_tracked_persons, string("/spencer/perception/tracked_persons"));
    private_node_handle_.param("tracked_persons_2d", topic_tracked_persons_2d, string("/spencer/perception/tracked_persons_2d"));

    // Create a subscriber.
    message_filters::Subscriber<spencer_tracking_msgs::TrackedPersons> subscriber_tracked_persons(n, topic_tracked_persons.c_str(), 10); subscriber_tracked_persons.unsubscribe();
    message_filters::Subscriber<spencer_tracking_msgs::TrackedPersons2d> subscriber_tracked_persons_2d(n, topic_tracked_persons_2d.c_str(), 10); subscriber_tracked_persons_2d.unsubscribe();

    //register callback + approx timing
    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                       boost::ref(subscriber_tracked_persons),
                                                       boost::ref(subscriber_tracked_persons_2d));
    sync_policies::ApproximateTime<spencer_tracking_msgs::TrackedPersons,
            spencer_tracking_msgs::TrackedPersons2d> MySyncPolicy(queue_size); //The real queue size for synchronisation is set here.

    const sync_policies::ApproximateTime<spencer_tracking_msgs::TrackedPersons,
           spencer_tracking_msgs::TrackedPersons2d> MyConstSyncPolicy = MySyncPolicy;

    Synchronizer< sync_policies::ApproximateTime<spencer_tracking_msgs::TrackedPersons,
            spencer_tracking_msgs::TrackedPersons2d> >
            sync(MyConstSyncPolicy, subscriber_tracked_persons, subscriber_tracked_persons_2d);
    sync.registerCallback(boost::bind(&callback, _1, _2));

    // Create a topic publisher
    private_node_handle_.param("tracked_persons2analyze", pub_topic_tracked_persons2analyze, string("/spencer/perception/tracked_persons2analyze"));
    pub_tracked_persons2analyze = n.advertise<spencer_tracking_msgs::TrackedPersons>(pub_topic_tracked_persons2analyze, 10, con_cb, con_cb);

    private_node_handle_.param("tracked_persons2analyze_2d", pub_topic_tracked_persons2analyze_2d, (pub_topic_tracked_persons2analyze + "_2d"));
    pub_tracked_persons2analyze_2d = n.advertise<spencer_tracking_msgs::TrackedPersons2d>(pub_topic_tracked_persons2analyze_2d, 10, con_cb, con_cb);

    ros::spin();
    return 0;
}
