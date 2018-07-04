#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string.h>

#include <message_filters/subscriber.h>

#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/PersonTrajectories.h"
#include "frame_msgs/PersonTrajectory.h"
#include "frame_msgs/PersonTrajectoryEntry.h"

#include <tf/tf.h>
#include <tf/transform_listener.h>

using namespace std;
using namespace message_filters;
using namespace frame_msgs;

ros::Publisher pub_person_trajectories;
ros::Publisher pub_selected_person_trajectory;
PersonTrajectories personTrajectories;

tf::TransformListener* listener;
string camera_frame;

vector<double> cartesianToPolar(geometry_msgs::Point point) {
    ROS_DEBUG("cartesianToPolar: Cartesian point: x: %f, y: %f, z %f", point.x, point.y, point.z);
    vector<double> output;
    double dist = sqrt(pow(point.x,2) + pow(point.z,2));
    double angle = atan2(point.z, point.x);
    output.push_back(dist);
    output.push_back(angle);
    ROS_DEBUG("cartesianToPolar: Polar point: distance: %f, angle: %f", dist, angle);
    return output;
}

void callback(const TrackedPersons::ConstPtr &tps)
{
    personTrajectories.header = tps->header;
    TrackedPerson tp;
    int selected_trajectory_idx = 0;
    float selected_trajectory_min_dist = 10000.0f;
    
    //for each trackedPerson tp in all trackedPersons tps
    for(int i = 0; i < tps->tracks.size(); i++){
        tp = tps->tracks.at(i);
        int t_id = tp.track_id;
        bool t_id_exists = false;
        //prepare personTrajectoryEntry pje
        PersonTrajectoryEntry pje;
        pje.pose = tp.pose;
        pje.twist = tp.twist;
        pje.age = tp.age;
        pje.is_occluded = tp.is_occluded;
        pje.detection_id = tp.detection_id;

        //compute distance to robot
        bool new_min_found = false;
        geometry_msgs::PointStamped distancePointStamped;
        geometry_msgs::PointStamped distancePointStampedCamera;
        vector<double> polCo;
        distancePointStamped.header.frame_id = tps->header.frame_id;
        distancePointStamped.header.stamp = ros::Time();
        distancePointStamped.point.x = pje.pose.pose.position.x;
        distancePointStamped.point.y = pje.pose.pose.position.y;
        distancePointStamped.point.z = pje.pose.pose.position.z;
        try {
            listener->waitForTransform(distancePointStamped.header.frame_id, camera_frame, ros::Time(), ros::Duration(1.0));
            listener->transformPoint(camera_frame, distancePointStamped, distancePointStampedCamera);
            //std::cout << "ID: " << t_id << " camX: " << distancePointStampedCamera.point.x << " camY: " << distancePointStampedCamera.point.y << " camZ: " << distancePointStampedCamera.point.z << std::endl;
            polCo = cartesianToPolar(distancePointStampedCamera.point);
            if(polCo.at(0) < selected_trajectory_min_dist){
                new_min_found = true;
                selected_trajectory_min_dist = polCo.at(0);
            }
        }
        catch(tf::TransformException ex) {
            ROS_WARN_THROTTLE(20.0, "Failed transform lookup in rwth_pedestrian_trajectories. Reason: %s. Message will re-appear within 20 seconds.", ex.what());
        }

        // loop through all existing ids in personTrajectories...
        for(int j = 0; j < personTrajectories.trajectories.size(); j++){
            if(personTrajectories.trajectories.at(j).track_id == t_id){
                // ...and add this personTrajectoryEntry pje to this id
                personTrajectories.trajectories.at(j).trajectory.push_back(pje);
                t_id_exists = true;
                if(new_min_found){
                    selected_trajectory_idx = j;
                }
                break;
            }
        }
        //new id?
        if (!t_id_exists){
            //new personTrajectory pj with one personTrajectoryEntry pje in personTrajectories
            PersonTrajectory pj;
            pj.track_id = t_id;
            pj.trajectory.push_back(pje);
            personTrajectories.trajectories.push_back(pj);
            if(new_min_found){
                selected_trajectory_idx = personTrajectories.trajectories.size()-1;
            }
        }
            
    }
    //publish all personTrajectories 
    pub_person_trajectories.publish(personTrajectories);

    // publish "selected" (right now: closest) trajectory on a seperate topic
    if(personTrajectories.trajectories.size()>0 && tps->tracks.size()>0){
        pub_selected_person_trajectory.publish(personTrajectories.trajectories.at(selected_trajectory_idx));
    }else{
        PersonTrajectory empty_pt;
        empty_pt.track_id = 0;
        pub_selected_person_trajectory.publish(empty_pt);
    }
    
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<TrackedPersons> &sub_tra){
    if(!pub_person_trajectories.getNumSubscribers()
        && !pub_selected_person_trajectory.getNumSubscribers()
    ) {
        ROS_DEBUG("Trajectories: No subscribers. Unsubscribing.");
        sub_tra.unsubscribe();
    } else {
        ROS_DEBUG("Trajectories: New subscribers. Subscribing.");
        sub_tra.subscribe();
    }
}

int main(int argc, char **argv)
{
    //init ROS
    ros::init(argc, argv, "pedestrian_trajectories");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string sub_topic_tracked_persons;
    string pub_topic_trajectories;
    string pub_topic_selected_trajectory;

    listener = new tf::TransformListener();


    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("tracked_persons", sub_topic_tracked_persons, string("/rwth_tracker/tracked_persons"));
    private_node_handle_.param("camera_frame", camera_frame, string("/camera/"));

    ROS_DEBUG("pedestrian_trajectories: Queue size for synchronisation is set to: %i", queue_size);

    // Create a subscriber.
    //ros::Subscriber sub = n.subscribe("chatter", 1000, callback);
 
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    message_filters::Subscriber<TrackedPersons> subscriber_tracks(n, sub_topic_tracked_persons.c_str(), 1); subscriber_tracks.unsubscribe();
    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback, boost::ref(subscriber_tracks));
    subscriber_tracks.registerCallback(boost::bind(&callback, _1));

    // Create a topic publisher
    private_node_handle_.param("person_trajectories", pub_topic_trajectories, string("/rwth_tracker/person_trajectories"));
    private_node_handle_.param("selected_person_trajectory", pub_topic_selected_trajectory, string("/rwth_tracker/selected_person_trajectory"));
    pub_person_trajectories = n.advertise<PersonTrajectories>(pub_topic_trajectories, 10, con_cb, con_cb);
    pub_selected_person_trajectory = n.advertise<PersonTrajectory>(pub_topic_selected_trajectory, 10, con_cb, con_cb);

    ros::spin();

    return 0;
}
