#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string.h>
#include <unordered_set>

#include <message_filters/subscriber.h>

#include "frame_msgs/TrackedPersons.h"
#include "frame_msgs/PersonTrajectories.h"
#include "frame_msgs/PersonTrajectory.h"
#include "frame_msgs/PersonTrajectoryEntry.h"
#include "frame_msgs/DetectedPersons.h"

#include <tf/tf.h>
#include <tf/transform_listener.h>

using namespace std;
using namespace message_filters;
using namespace frame_msgs;

ros::Publisher pub_person_trajectories;
ros::Publisher pub_selected_person_trajectory;
ros::Publisher pub_selected_helper_vis;
ros::Publisher pub_potential_helpers_vis;
PersonTrajectories personTrajectories;

tf::TransformListener* listener;
string camera_frame;

bool keep = true; //if true, a selected ID is kept, even if others fulfill the criteria better
bool strict = false; //if true, a selected ID needs to fulfill the criteria all the time
bool remember = true; //TODO: if true, a selected ID had to fulfill the criteria at one point in time and can be selected later
int last_selected_person_id = -1;
unordered_set<int> pastHelperIds;
unordered_set<int> blacklistedHelperIds;

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
    int selected_trajectory_idx = -1;
    float selected_trajectory_min_dist = 10000.0f;
    float max_dist = 3.0f; //maximum distance to be selected

    DetectedPersons potentialHelpersVis;
    potentialHelpersVis.header = tps->header;
    bool last_person_selected_again = false;
    
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
        bool is_potential_helper = false;
        bool is_best_helper = false;
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
            polCo = cartesianToPolar(distancePointStampedCamera.point);
            if(polCo.at(0) <= max_dist || (remember && pastHelperIds.count(t_id)>0 && !strict)){
                // fulfills criterion, add to potential helpers
                DetectedPerson potentialHelper;
                potentialHelper.confidence = 0.5;
                potentialHelper.pose.pose.position = tp.pose.pose.position;
                potentialHelpersVis.detections.push_back(potentialHelper);
                is_potential_helper = true;
                if(polCo.at(0) < selected_trajectory_min_dist){
                    // fulfills criterion best, set new min
                    is_best_helper = true;
                    selected_trajectory_min_dist = polCo.at(0);
                }
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
                if(keep && last_selected_person_id==t_id && (!strict || is_potential_helper)){
                    //std::cout << "last selected person found! it is " << t_id << std::endl;
                    last_person_selected_again = true;
                    selected_trajectory_idx = j;
                    //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
                }else if(is_best_helper && !last_person_selected_again){
                    selected_trajectory_idx = j;
                    //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
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
            if(is_best_helper && !last_person_selected_again){
                //std::cout << "new min found and last selected person was not found (or should not be kept) " << std::endl;
                selected_trajectory_idx = personTrajectories.trajectories.size()-1;
                //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
            }
        }
            
    }

    //publish all personTrajectories and the visualization of potential helpers
    pub_person_trajectories.publish(personTrajectories);
    pub_potential_helpers_vis.publish(potentialHelpersVis);

    // publish "selected" (right now: closest) trajectory on a seperate topic
    if(selected_trajectory_idx!=-1 && personTrajectories.trajectories.size()>0 && tps->tracks.size()>0){
        PersonTrajectory selectedPersonTrajectory = personTrajectories.trajectories.at(selected_trajectory_idx);
        pub_selected_person_trajectory.publish(selectedPersonTrajectory);
        last_selected_person_id = selectedPersonTrajectory.track_id;
        pastHelperIds.insert(last_selected_person_id);
        //std::cout << "new last ID: " << last_selected_person_id << std::endl;
        // publish a DetectedPersonsArray of this for visualization purposes1
        DetectedPersons selectedHelperVis;
        selectedHelperVis.header = tps->header;
        DetectedPerson currentSelectedPersonSolo;
        currentSelectedPersonSolo.confidence = 1.0;
        currentSelectedPersonSolo.pose.pose.position = selectedPersonTrajectory.trajectory.at(selectedPersonTrajectory.trajectory.size()-1).pose.pose.position;
        selectedHelperVis.detections.push_back(currentSelectedPersonSolo);
        pub_selected_helper_vis.publish(selectedHelperVis);
    }else{
        ROS_DEBUG("no person trajectory selected");
        //PersonTrajectory empty_pt;
        //empty_pt.track_id = 0;
        //pub_selected_person_trajectory.publish(empty_pt);
        DetectedPersons currentSelectedPersonVis;
        currentSelectedPersonVis.header = tps->header;
        pub_selected_helper_vis.publish(currentSelectedPersonVis);
    }
    
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<TrackedPersons> &sub_tra){
    if(!pub_person_trajectories.getNumSubscribers()
        && !pub_selected_person_trajectory.getNumSubscribers()
        && !pub_selected_helper_vis.getNumSubscribers()
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
    string pub_topic_selected_helper_vis;
    string pub_topic_potential_helpers_vis;

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
    private_node_handle_.param("selected_helper_vis", pub_topic_selected_helper_vis, string("/rwth_tracker/selected_helper_vis"));
    private_node_handle_.param("potential_helpers_vis", pub_topic_potential_helpers_vis, string("/rwth_tracker/potential_helpers_vis"));
    pub_person_trajectories = n.advertise<PersonTrajectories>(pub_topic_trajectories, 10, con_cb, con_cb);
    pub_selected_person_trajectory = n.advertise<PersonTrajectory>(pub_topic_selected_trajectory, 10, con_cb, con_cb);
    pub_selected_helper_vis = n.advertise<DetectedPersons>(pub_topic_selected_helper_vis, 10, con_cb, con_cb);
    pub_potential_helpers_vis = n.advertise<DetectedPersons>(pub_topic_potential_helpers_vis, 10, con_cb, con_cb);

    ros::spin();

    return 0;
}
