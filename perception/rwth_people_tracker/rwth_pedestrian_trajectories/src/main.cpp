#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string.h>

#include <message_filters/subscriber.h>

#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/PersonTrajectories.h"
#include "frame_msgs/PersonTrajectory.h"
#include "frame_msgs/PersonTrajectoryEntry.h"

using namespace std;
using namespace message_filters;
using namespace frame_msgs;

ros::Publisher pub_person_trajectories;
PersonTrajectories personTrajectories;


void callback(const TrackedPersons::ConstPtr &tps)
{
    ROS_INFO("Trajectories cb entered!");

    personTrajectories.header = tps->header;
    TrackedPerson tp;
    
    //for all tp in tps
    for(int i = 0; i < tps->tracks.size(); i++){
        tp = tps->tracks.at(i);
        int t_id = tp.track_id;
        bool t_id_exists = false;
        PersonTrajectoryEntry pje;
        pje.pose = tp.pose;
        pje.twist = tp.twist;
        pje.age = tp.age;
        pje.is_occluded = tp.is_occluded;
        pje.detection_id = tp.detection_id;

        for(int j = 0; j < personTrajectories.trajectories.size(); j++){
            if(personTrajectories.trajectories.at(j).track_id == t_id){
                personTrajectories.trajectories.at(j).trajectory.push_back(pje);
                t_id_exists = true;
                break;
            }
        }
        //new id?
        if (!t_id_exists){
            //new pj with one new pje in pjs
            PersonTrajectory pj;
            pj.track_id = t_id;
            pj.trajectory.push_back(pje);
            personTrajectories.trajectories.push_back(pj);  
        }
        
    }
    //publish pjs 
    pub_person_trajectories.publish(personTrajectories);
    
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(ros::NodeHandle &n, message_filters::Subscriber<TrackedPersons> &sub_tra, string topic, int qs){
    if(!pub_person_trajectories.getNumSubscribers()
    ) {
        ROS_DEBUG("Trajectories: No subscribers. Unsubscribing.");
        sub_tra.unsubscribe();
    } else {
        ROS_DEBUG("Trajectories: New subscribers. Subscribing.");
        //sub_tra = n.subscribe(topic.c_str(), qs, &callback);
        sub_tra.subscribe();
    }
}

int main(int argc, char **argv)
{
    //init ROS
    ros::init(argc, argv, "listener");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string sub_topic_tracked_persons;
    string pub_topic_trajectories;


    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("tracked_persons", sub_topic_tracked_persons, string("/rwth_tracker/tracked_persons"));

    ROS_DEBUG("pedestrian_trajectories: Queue size for synchronisation is set to: %i", queue_size);

    // Create a subscriber.
    //ros::Subscriber sub = n.subscribe("chatter", 1000, callback);
 
    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    message_filters::Subscriber<TrackedPersons> subscriber_tracks(n, sub_topic_tracked_persons.c_str(), 1); subscriber_tracks.unsubscribe();
    //ros::Subscriber subscriber_tracks;
    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback, boost::ref(n), boost::ref(subscriber_tracks), sub_topic_tracked_persons, queue_size);

    // Create a topic publisher
    private_node_handle_.param("person_trajectories", pub_topic_trajectories, string("/rwth_tracker/person_trajectories"));
    pub_person_trajectories = n.advertise<PersonTrajectories>(pub_topic_trajectories, 10, con_cb, con_cb);

    ros::spin();

    return 0;
}
