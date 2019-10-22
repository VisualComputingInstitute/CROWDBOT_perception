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
#include "std_msgs/Bool.h"

#include <tf/tf.h>
#include <tf/transform_listener.h>

#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>

using namespace std;
using namespace message_filters;
using namespace frame_msgs;

ros::Publisher pub_person_trajectories;
ros::Publisher pub_selected_helper;
ros::Publisher pub_selected_helper_vis;
ros::Publisher pub_potential_helpers;
ros::Publisher pub_potential_helpers_vis;
ros::Publisher pub_helper_search_status;
ros::Publisher pub_deselect_ack;
ros::Publisher pub_new_search_ack;
ros::Publisher pub_helper_selected;
PersonTrajectories personTrajectories;

tf::TransformListener* listener;
string camera_frame;

//const char *homedir = getpwuid(getuid())->pw_dir;
string homedir = getenv("HOME");

std::ofstream outfile;

bool keep; //if true, a selected ID is kept, even if others fulfill the criteria better
bool strict; //if true, a selected ID needs to fulfill the criteria all the time
bool remember; //if true, a once selected ID will always be considered as potential helper, if it fulfills the criteria (or strict is false)
int last_selected_person_id = -1;
std::vector<float> last_selected_person_emb_vec;
unordered_set<int> past_helper_ids;
unordered_set<int> blacklistedHelperIds;

bool is_helper_selected_mem = false;

int trajectory_max_length = 50;

bool new_search_invoked = false;
bool stop_selection = false;

double helper_reid_thresh;

double l2_norm(vector<float> const& u, vector<float> const& v) {
    if(u.size() != v.size()){
        //cout << "error when computing norm of vectors in helper selection: u and v are not of the same size, u: " << u.size() << ", v: "  << v.size() << endl;
        return 999.0;
    }
    double accum = 0.;
    for (int i = 0; i < u.size(); ++i) {
        accum += (u[i]-v[i]) * (u[i]-v[i]);
    }
    //cout << "success, norm is " << sqrt(accum) << endl;
    return sqrt(accum);
}

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

void callback_newSearch(const std_msgs::Bool::ConstPtr &newSearch)
{
    outfile.open(homedir+"/log/helper_callback_debug.txt", std::ios_base::app);
    outfile << ros::Time::now() << ": enter new_search callback with data=" << (newSearch->data? "True" : "False") <<"\n";
    new_search_invoked = true;
    if(newSearch->data && last_selected_person_id!=-1){
        blacklistedHelperIds.insert(last_selected_person_id);
    }
    stop_selection = false;
    //cout << "new search invoked by blacklisting current helper with ID " << last_selected_person_id << endl;
    outfile << ros::Time::now() << ": exit new_search callback" <<  "\n";
    outfile.close();
    std_msgs::Bool new_search_ack = std_msgs::Bool();
    new_search_ack.data = true;
    pub_new_search_ack.publish(new_search_ack);
    is_helper_selected_mem = false;

}

void callback_resetHelperBlacklist(const std_msgs::Bool::ConstPtr &resetBlacklist){
    outfile.open(homedir+"/log/helper_callback_debug.txt", std::ios_base::app);
    outfile << ros::Time::now() << ": enter resetBlacklist callback with data=" << (resetBlacklist->data? "True" : "False") <<  "\n";
   blacklistedHelperIds.clear();
   outfile << ros::Time::now() << ": exit resetBlacklist callback" <<  "\n";
   outfile.close();
}

void callback_stopHelperSelection(const std_msgs::Bool::ConstPtr &stop_helper_selection){
    outfile.open(homedir+"/log/helper_callback_debug.txt", std::ios_base::app);
    outfile << ros::Time::now() << ": enter stopHelperSelection callback with data=" << (stop_helper_selection->data? "True" : "False") <<  "\n";
   stop_selection = true;
   if(stop_helper_selection->data && last_selected_person_id!=-1){
       blacklistedHelperIds.insert(last_selected_person_id);
   }
   new_search_invoked = false;
   last_selected_person_id = -1;
   outfile << ros::Time::now() << ": exit stopHelperSelection callback" <<  "\n";
   outfile.close();
   std_msgs::Bool deselect_ack = std_msgs::Bool();
   deselect_ack.data = true;
   pub_deselect_ack.publish(deselect_ack);
   is_helper_selected_mem = true;
}

void callback(const TrackedPersons::ConstPtr &tps)
{
    const float max_dist = 5.0f; //maximum distance to be selected

    personTrajectories.header = tps->header;
    int selected_trajectory_idx = -1;
    float selected_trajectory_min_dist = 10000.0f;
    bool last_person_selected_again = false;
    double min_emb_dist = 999.0;
    std::vector<float> curr_track_emb_vec;

    DetectedPersons potentialHelpersVis;
    potentialHelpersVis.header = tps->header;
    PersonTrajectories potentialHelpers;
    potentialHelpers.header = tps->header;

    //for each trackedPerson tp in all trackedPersons tps
    for (int i = 0; i < tps->tracks.size(); i++)
    {
        const TrackedPerson& tp = tps->tracks.at(i);
        const int t_id = tp.track_id;

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

        const bool blacklisted = blacklistedHelperIds.count(t_id)>0; //not added as potential helper, trajectory is updated anyway

        try
        {
            geometry_msgs::PointStamped distancePointStamped, distancePointStampedCamera;
            distancePointStamped.header.frame_id = tps->header.frame_id;
            distancePointStamped.header.stamp = ros::Time();
            distancePointStamped.point.x = pje.pose.pose.position.x;
            distancePointStamped.point.y = pje.pose.pose.position.y;
            distancePointStamped.point.z = pje.pose.pose.position.z;

            listener->waitForTransform(distancePointStamped.header.frame_id, camera_frame, ros::Time(), ros::Duration(1.0));
            listener->transformPoint(camera_frame, distancePointStamped, distancePointStampedCamera);

            vector<double> polCo;
            polCo = cartesianToPolar(distancePointStampedCamera.point);
            const double dist_to_cam = polCo.at(0);

            const bool is_close = dist_to_cam <= max_dist;
            const bool is_prev_helper = remember && past_helper_ids.count(t_id)>0 && !strict;

            if (!blacklisted && (is_close || is_prev_helper))
            {
                // fulfills criterion, add to potential helpers
                DetectedPerson potentialHelper;
                potentialHelper.confidence = 0.5;
                potentialHelper.pose.pose.position = tp.pose.pose.position;
                potentialHelpersVis.detections.push_back(potentialHelper);
                is_potential_helper = true;

                if (dist_to_cam < selected_trajectory_min_dist)
                {
                    is_best_helper = true;
                    selected_trajectory_min_dist = dist_to_cam;
                }
            }
        }
        catch (tf::TransformException ex)
        {
            ROS_WARN_THROTTLE(20.0, "Failed transform lookup in rwth_pedestrian_trajectories. Reason: %s. Message will re-appear within 20 seconds.", ex.what());
        }

        // loop through all existing ids in personTrajectories...
        int t_id_found_at = -1;
        for (int j = 0; j < personTrajectories.trajectories.size(); j++)
        {
            if (personTrajectories.trajectories.at(j).track_id != t_id)
            {
                continue;
            }

            t_id_found_at = j;

            // ...and add this personTrajectoryEntry pje to this id
            // limit person Trajectory entries to trajectory_max_length
            if(personTrajectories.trajectories.at(j).trajectory.size() > trajectory_max_length)
            {
                std::rotate(personTrajectories.trajectories.at(j).trajectory.begin(),
                            personTrajectories.trajectories.at(j).trajectory.begin()+1,
                            personTrajectories.trajectories.at(j).trajectory.end());
                personTrajectories.trajectories.at(j).trajectory.at(personTrajectories.trajectories.at(j).trajectory.size()-1) = pje;
            }
            else
            {
                personTrajectories.trajectories.at(j).trajectory.push_back(pje);
            }

            // update reid embedding vector
            personTrajectories.trajectories.at(j).embed_vector = tp.embed_vector;
            if (!blacklisted && keep && last_selected_person_id==t_id && (!strict || is_potential_helper))
            {
                //std::cout << "last selected person found! it is " << t_id << std::endl;
                last_person_selected_again = true;
                selected_trajectory_idx = j;
                //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
            }
            else if (is_best_helper && !last_person_selected_again && !blacklisted && new_search_invoked)
            {
                selected_trajectory_idx = j;
                //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
            }

            if (is_potential_helper)
            {
                potentialHelpers.trajectories.push_back(personTrajectories.trajectories.at(j));
            }

            break;
        }

        //new id?
        if (t_id_found_at == -1)
        {
            //new personTrajectory pj with one personTrajectoryEntry pje in personTrajectories
            PersonTrajectory pj;
            pj.track_id = t_id;
            pj.embed_vector = tp.embed_vector;
            pj.trajectory.push_back(pje);
            personTrajectories.trajectories.push_back(pj);
            if (is_best_helper && !last_person_selected_again && !blacklisted && new_search_invoked)
            {
                //std::cout << "new min found and last selected person was not found (or should not be kept) " << std::endl;
                selected_trajectory_idx = personTrajectories.trajectories.size() - 1;
                //std::cout << "set selected index to: " << selected_trajectory_idx << std::endl;
            }
            t_id_found_at = personTrajectories.trajectories.size() - 1;
            if (is_potential_helper)
            {
                potentialHelpers.trajectories.push_back(personTrajectories.trajectories.back());
            }
        }

        // compute reid embedding distance to last selected person and set new min
        // (only neccessary if last helper has not been found, if there was any yet + not blacklisted)
        if (last_selected_person_id != -1 && !last_person_selected_again && !blacklisted)
        {
            curr_track_emb_vec = tp.embed_vector;
            const double emb_dist = l2_norm(last_selected_person_emb_vec, curr_track_emb_vec);
            if (emb_dist < helper_reid_thresh && emb_dist < min_emb_dist && helper_reid_thresh != 0)
            {
                // if embedding distance to existing trajectory low enough (+last helper is/was not found), helper has probably switched ID
                min_emb_dist = emb_dist;
                selected_trajectory_idx = t_id_found_at;
            }
        }
    }  // for (int i = 0; i < tps->tracks.size(); i++)

    //publish all personTrajectories and the visualization of potential helpers
    personTrajectories.header.stamp = ros::Time::now();
    pub_person_trajectories.publish(personTrajectories);
    pub_potential_helpers.publish(potentialHelpers);
    pub_potential_helpers_vis.publish(potentialHelpersVis);

    // search status topic
    std_msgs::Bool status = std_msgs::Bool();
    if(new_search_invoked){
        status.data = true;
        pub_helper_search_status.publish(status);
    }else{
        status.data = false;
        pub_helper_search_status.publish(status);
    }

    // publish "selected" (right now: closest) trajectory on a seperate topic
    if(selected_trajectory_idx!=-1 && personTrajectories.trajectories.size()>0 && tps->tracks.size()>0 && !stop_selection){
        PersonTrajectory selectedPersonTrajectory = personTrajectories.trajectories.at(selected_trajectory_idx);
        pub_selected_helper.publish(selectedPersonTrajectory);
        last_selected_person_id = selectedPersonTrajectory.track_id;
        last_selected_person_emb_vec = selectedPersonTrajectory.embed_vector;
        past_helper_ids.insert(last_selected_person_id);
        //std::cout << "new last ID: " << last_selected_person_id << std::endl;
        // publish a DetectedPersonsArray of this for visualization purposes1
        DetectedPersons selectedHelperVis;
        selectedHelperVis.header = tps->header;
        DetectedPerson currentSelectedPersonSolo;
        currentSelectedPersonSolo.confidence = 1.0;
        currentSelectedPersonSolo.pose.pose.position = selectedPersonTrajectory.trajectory.at(selectedPersonTrajectory.trajectory.size()-1).pose.pose.position;
        selectedHelperVis.detections.push_back(currentSelectedPersonSolo);
        pub_selected_helper_vis.publish(selectedHelperVis);
        new_search_invoked = false;
        if (!is_helper_selected_mem){
            std_msgs::Bool is_helper_selected = std_msgs::Bool();
            is_helper_selected.data = true;
            pub_helper_selected.publish(is_helper_selected);
            is_helper_selected_mem = true;
        }
    }else{
        ROS_DEBUG("no person trajectory selected"); //possible options: publish empty, last, nothing (currently nothing)
        //PersonTrajectory empty_pt;
        //empty_pt.track_id = 0;
        //pub_selected_person_trajectory.publish(empty_pt);
        if (is_helper_selected_mem){
            std_msgs::Bool is_helper_selected = std_msgs::Bool();
            is_helper_selected.data = false;
            pub_helper_selected.publish(is_helper_selected);
            is_helper_selected_mem = false;
        }

        //always publish empty vis to avoid ghost vis
        DetectedPersons currentSelectedPersonVis;
        currentSelectedPersonVis.header = tps->header;
        pub_selected_helper_vis.publish(currentSelectedPersonVis);
    }

}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<TrackedPersons> &sub_tra){
    if(!pub_person_trajectories.getNumSubscribers()
        && !pub_selected_helper.getNumSubscribers()
        && !pub_selected_helper_vis.getNumSubscribers()
        && !pub_potential_helpers.getNumSubscribers()
        && !pub_potential_helpers_vis.getNumSubscribers()
        && !pub_helper_search_status.getNumSubscribers()
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
    string sub_topic_new_search;
    string sub_topic_reset_helper_blacklist;
    string sub_topic_stop_helper_selection;
    string pub_topic_trajectories;
    string pub_topic_selected_helper;
    string pub_topic_potential_helpers;
    string pub_topic_selected_helper_vis;
    string pub_topic_potential_helpers_vis;
    string pub_topic_helper_search_status;
    string pub_topic_helper_selected;

    listener = new tf::TransformListener();

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("tracked_persons", sub_topic_tracked_persons, string("/rwth_tracker/tracked_persons"));
    private_node_handle_.param("get_new_helper", sub_topic_new_search, string("/rwth_tracker/get_new_helper"));
    private_node_handle_.param("reset_helper_blacklist", sub_topic_reset_helper_blacklist, string("/rwth_tracker/reset_helper_blacklist"));
    private_node_handle_.param("stop_helper_selection", sub_topic_stop_helper_selection, string("/rwth_tracker/stop_helper_selection"));
    private_node_handle_.param("camera_frame", camera_frame, string("/camera/"));
    // helper selection options
    private_node_handle_.param("keep", keep, true);
    private_node_handle_.param("strict", strict, false);
    private_node_handle_.param("remember", remember, true);
    //threshold to reidentify helper
    private_node_handle_.param("helper_reid_thresh", helper_reid_thresh, double(50));

    ROS_DEBUG("pedestrian_trajectories: Queue size for synchronisation is set to: %i", queue_size);

    // Create a subscriber.
    //ros::Subscriber sub = n.subscribe("chatter", 1000, callback);

    // Set queue size to 1 because generating a queue here will only pile up images and delay the output by the amount of queued images
    // The immediate unsubscribe is necessary to start without subscribing to any topic because message_filters does nor allow to do it another way.
    message_filters::Subscriber<TrackedPersons> subscriber_tracks(n, sub_topic_tracked_persons.c_str(), 1); subscriber_tracks.unsubscribe();
    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback, boost::ref(subscriber_tracks));
    subscriber_tracks.registerCallback(boost::bind(&callback, _1));
    message_filters::Subscriber<std_msgs::Bool> subscriber_new_search(n, sub_topic_new_search.c_str(), 1); subscriber_new_search.unsubscribe();
    subscriber_new_search.registerCallback(boost::bind(&callback_newSearch, _1));
    subscriber_new_search.subscribe();
    message_filters::Subscriber<std_msgs::Bool> subscriber_reset_helper_blacklist(n, sub_topic_reset_helper_blacklist.c_str(), 1); subscriber_reset_helper_blacklist.unsubscribe();
    subscriber_reset_helper_blacklist.registerCallback(boost::bind(&callback_resetHelperBlacklist, _1));
    subscriber_reset_helper_blacklist.subscribe();
    message_filters::Subscriber<std_msgs::Bool> subscriber_stop_helper_selection(n, sub_topic_stop_helper_selection.c_str(), 1); subscriber_stop_helper_selection.unsubscribe();
    subscriber_stop_helper_selection.registerCallback(boost::bind(&callback_stopHelperSelection, _1));
    subscriber_stop_helper_selection.subscribe();

    // Create a topic publisher
    private_node_handle_.param("person_trajectories", pub_topic_trajectories, string("/rwth_tracker/person_trajectories"));
    private_node_handle_.param("selected_helper", pub_topic_selected_helper, string("/rwth_tracker/selected_helper"));
    private_node_handle_.param("potential_helpers", pub_topic_potential_helpers, string("/rwth_tracker/potential_helpers"));
    private_node_handle_.param("selected_helper_vis", pub_topic_selected_helper_vis, string("/rwth_tracker/selected_helper_vis"));
    private_node_handle_.param("potential_helpers_vis", pub_topic_potential_helpers_vis, string("/rwth_tracker/potential_helpers_vis"));
    private_node_handle_.param("helper_search_status", pub_topic_helper_search_status, string("/rwth_tracker/helper_search_status"));
    private_node_handle_.param("helper_selected", pub_topic_helper_selected, string("/rwth_tracker/helper_selected"));
    pub_person_trajectories = n.advertise<PersonTrajectories>(pub_topic_trajectories, 1, con_cb, con_cb);
    pub_selected_helper = n.advertise<PersonTrajectory>(pub_topic_selected_helper, 1, con_cb, con_cb);
    pub_potential_helpers = n.advertise<PersonTrajectories>(pub_topic_potential_helpers, 1, con_cb, con_cb);
    pub_selected_helper_vis = n.advertise<DetectedPersons>(pub_topic_selected_helper_vis, 1, con_cb, con_cb);
    pub_potential_helpers_vis = n.advertise<DetectedPersons>(pub_topic_potential_helpers_vis, 1, con_cb, con_cb);
    pub_helper_search_status = n.advertise<std_msgs::Bool>(pub_topic_helper_search_status, 1, con_cb, con_cb);
    pub_deselect_ack = n.advertise<std_msgs::Bool>(sub_topic_stop_helper_selection + "_ACK", 1, con_cb, con_cb);
    pub_new_search_ack = n.advertise<std_msgs::Bool>(sub_topic_new_search + "_ACK", 1, con_cb, con_cb);
    pub_helper_selected = n.advertise<std_msgs::Bool>(pub_topic_helper_selected, 1, con_cb, con_cb);

    outfile.open(homedir+"/log/helper_callback_debug.txt", std::ofstream::out | std::ofstream::trunc);
    outfile.close();

    ros::spin();

    return 0;
}
