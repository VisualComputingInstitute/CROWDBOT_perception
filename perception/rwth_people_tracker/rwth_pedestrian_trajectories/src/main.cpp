#include "ros/ros.h"
#include "std_msgs/String.h"
#include <string.h>
#include <unordered_set>
#include <nav_msgs/OccupancyGrid.h>
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

using namespace frame_msgs;

class HelperSearchInterface
{
public:
    HelperSearchInterface()
    {
        const std::string home = std::getenv("HOME");
        const std::string f = home + "/log/helper_callback_debug.txt";
        ofs.open(f, std::ofstream::out | std::ofstream::trunc);
    }

    ~HelperSearchInterface()
    {
        ofs.close();
    }

    void addBlacklist(const int idx)
    {
        if (idx >= 0)
            id_blacklist_.insert(idx);
    }

    bool checkBlacklist(const int idx)
    {
        for (const int val : id_blacklist_)
        {
            if (idx == val)
                return true;
        }
        return false;
    }

    void clearBlacklist()
    {
        id_blacklist_.clear();
    }

    void publishSuccess(const bool s)
    {
        std_msgs::Bool msg = std_msgs::Bool();
        msg.data = s;
        success_pub.publish(msg);
    }

    void publishStatus()
    {
        std_msgs::Bool msg = std_msgs::Bool();
        msg.data = active;
        status_pub.publish(msg);
    }

    //
    ros::Publisher start_ack_pub;
    ros::Publisher stop_ack_pub;
    ros::Publisher status_pub;
    ros::Publisher success_pub;

    // Config
    bool active = false;   // True to enable helper selection
    bool use_map = true;  // True to take into account of obstacle when selecting helpers, using an existing map.
    bool strict = false;   // True if a selected ID needs to fulfill the criteria all the time.
    // bool remember = true;  // True then a once selected ID will always be considered as potential helper, if it fulfills the criteria (or strict is false)
    bool keep = true;  // True will give priority to last selected person for helper.

    int last_selected_person_id = -1;  //
    std::ofstream ofs;  // Output stream for debug

private:
    std::unordered_set<int> id_blacklist_;
};

HelperSearchInterface hs_;


ros::Publisher person_trajectories_pub_;
ros::Publisher selected_helper_pub_;
ros::Publisher selected_helper_viz_pub_;
ros::Publisher good_helpers_pub_;
ros::Publisher good_helpers_viz_pub_;

PersonTrajectories person_trajectories_;

tf::TransformListener* tf_;
std::string camera_frame_;

int trajectory_max_length_ = 50;
int number_past_track_for_occlusion_ = 3;
float max_dist_ = 5.0;  // Maximum distance for a helper

std::shared_ptr<nav_msgs::OccupancyGrid> map_ptr_(nullptr);

void callback_newSearch(const std_msgs::Bool::ConstPtr& msg)
{
    hs_.ofs << ros::Time::now()
            << ": enter new_search callback with data="
            << (msg->data ? "True" : "False")
            << "\n";
    hs_.active = true;

    if (msg->data)
    {
        hs_.addBlacklist(hs_.last_selected_person_id);
    }

    // stop_selection = false;
    std_msgs::Bool new_search_ack = std_msgs::Bool();
    new_search_ack.data = true;
    hs_.start_ack_pub.publish(new_search_ack);
    // is_helper_selected_mem = false;

    hs_.ofs << ros::Time::now()
            << ": exit new_search callback"
            <<  "\n";
}

void callback_resetHelperBlacklist(const std_msgs::Bool::ConstPtr& msg)
{
    hs_.ofs << ros::Time::now()
            << ": enter resetBlacklist callback with data="
            << (msg->data ? "True" : "False")
            << "\n";
    hs_.clearBlacklist();
    hs_.ofs << ros::Time::now()
            << ": exit resetBlacklist callback"
            <<  "\n";
}

void callback_stopHelperSelection(const std_msgs::Bool::ConstPtr& msg)
{
    hs_.ofs << ros::Time::now()
            << ": enter stopHelperSelection callback with data="
            << (msg->data ? "True" : "False")
            << "\n";
    hs_.active = false;
    if (msg->data)
    {
        hs_.addBlacklist(hs_.last_selected_person_id);
    }
    hs_.last_selected_person_id = -1;

    hs_.ofs << ros::Time::now()
            << ": exit stopHelperSelection callback"
            << "\n";

   std_msgs::Bool deselect_ack = std_msgs::Bool();
   deselect_ack.data = true;
   hs_.stop_ack_pub.publish(deselect_ack);
}

bool checkReachability(const geometry_msgs::PointStamped& start_pt,
                       const geometry_msgs::PointStamped& goal_pt)
{
    if (!map_ptr_)
    {
        ROS_WARN_THROTTLE(20.0, "No map. Ignore map check.");
        return true;
    }

    if (!hs_.use_map)
        return true;

    const std::string map_frame = map_ptr_->header.frame_id;
    geometry_msgs::PointStamped start_pt_map, goal_pt_map;
    try
    {
        tf_->waitForTransform(start_pt.header.frame_id, map_frame, start_pt.header.stamp, ros::Duration(1.0));
        tf_->transformPoint(map_frame, start_pt, start_pt_map);
        tf_->transformPoint(map_frame, goal_pt, goal_pt_map);
    }
    catch (tf::TransformException ex)
    {
        ROS_WARN_THROTTLE(20.0, "Failed transform lookup in rwth_pedestrian_trajectories. Reason: %s. Message will re-appear within 20 seconds. Ignore map check.", ex.what());
        return true;
    }

    const double map_x = map_ptr_->info.origin.position.x;
    const double map_y = map_ptr_->info.origin.position.y;
    const int map_width = map_ptr_->info.width;
    const int map_height = map_ptr_->info.height;
    const double map_resolution = map_ptr_->info.resolution;

    // std::cout << "Start pose (x, y): " << start_stamp_map.point.x << " " << start_stamp_map.point.y << " ";
    // std::cout << "Goal pose (x, y): " << goal_stamp_map.point.x << " " << goal_stamp_map.point.y << " ";
    // std::cout << "Map pose (x, y): " << map_x << " " << map_y << std::endl;

    const int x_idx_start = (start_pt_map.point.x - map_x) / map_resolution;
    const int x_idx_goal = (goal_pt_map.point.x - map_x) / map_resolution;
    const int y_idx_start = (start_pt_map.point.y - map_y) / map_resolution;
    const int y_idx_goal = (goal_pt_map.point.y - map_y) / map_resolution;

    const int x0 = std::max(std::min(x_idx_start, x_idx_goal), 0);
    const int x1 = std::min(std::max(x_idx_start, x_idx_goal), map_width);
    const int y0 = std::max(std::min(y_idx_start, y_idx_goal), 0);
    const int y1 = std::min(std::max(y_idx_start, y_idx_goal), map_height);
    const int xdiff = x1 - x0;
    const int ydiff = y1 - y0;

    // std::cout << "x0 " << x0 << " y0 " << y0
    //           << " x1 " << x1 << " y1 " << y1 << std::endl;

    // edge case: person and robot are in same map cell
    if (xdiff == 0 && ydiff == 0)
    {
        return true;
    }

    // create search indices along line
    std::vector<int> map_indices;
    map_indices.reserve(10000);
    if (ydiff > xdiff)
    {
        const double x_over_y = xdiff / ydiff;
        for (int y = y0; y < y1; y++)
        {
            const int x = x_over_y * (y - y0) + x0;
            map_indices.push_back(y * map_width + x);
        }
    }
    else
    {
        const double y_over_x = ydiff / xdiff;
        int obstacle_count = 0;
        for (int x = x0; x < x1; x++)
        {
            const int y = y_over_x * (x - x0) + y0;
            map_indices.push_back(y * map_width + x);
        }
    }
    // std::cout << "number of map cells: " << map_indices.size() << std::endl;

    // check grid occupancy along line
    int obstacle_count = 0;
    const auto& map_data = map_ptr_->data;
    for (int idx : map_indices)
    {
        const int val = static_cast<int>(map_data[idx]);
        // std::cout << val << std::endl;
        if (val > 50)
        {
            obstacle_count++;
        }

        if (obstacle_count > 0)
        {
            return false;
        }
    }

    return true;
}

int updateTrajectories(const TrackedPerson& tp)
{
    const int tp_id = tp.track_id;
    PersonTrajectoryEntry pje;
    pje.pose = tp.pose;
    pje.twist = tp.twist;
    pje.age = tp.age;
    pje.is_occluded = tp.is_occluded;
    pje.detection_id = tp.detection_id;

    // Find idx of the corresponding trajectory
    int idx = person_trajectories_.trajectories.size() - 1;
    while (idx >= 0 &&
           person_trajectories_.trajectories.at(idx).track_id != tp_id)
    {
        idx--;
    }

    // Update corresponding trajectory with new tracking evidance
    if (idx >= 0)
    {
        // Extend trajectory
        auto& matching_trj = person_trajectories_.trajectories.at(idx).trajectory;
        if (matching_trj.size() > trajectory_max_length_)
        {
            std::rotate(matching_trj.begin(), matching_trj.begin() + 1, matching_trj.end());
            matching_trj.pop_back();
        }
        matching_trj.push_back(pje);

        // Update reid embedding vector
        person_trajectories_.trajectories.at(idx).embed_vector = tp.embed_vector;

        return idx;
    }

    // If no corresponding trajectory found, create a new trajectory.
    PersonTrajectory pj;
    pj.track_id = tp_id;
    pj.embed_vector = tp.embed_vector;
    pj.trajectory.push_back(pje);
    person_trajectories_.trajectories.push_back(pj);

    return person_trajectories_.trajectories.size() - 1;
}

bool getRobotLocation(geometry_msgs::PointStamped& pt_stamped_target)
{
    // Use camera center as an approximate for robot location.
    geometry_msgs::PointStamped pt_stamped;
    pt_stamped.point.x = 0.0;
    pt_stamped.point.y = 0.0;
    pt_stamped.point.z = 0.0;

    const std::string target_frame = person_trajectories_.header.frame_id;
    const ros::Time& time = person_trajectories_.header.stamp;

    try
    {
        tf_->waitForTransform(target_frame, camera_frame_, time, ros::Duration(1.0));
        tf_->transformPoint(target_frame, pt_stamped, pt_stamped_target);
    }
    catch (tf::TransformException ex)
    {
        ROS_WARN_THROTTLE(20.0, "Failed transform lookup in rwth_pedestrian_trajectories. Reason: %s. Message will re-appear within 20 seconds.", ex.what());
        return false;
    }

    pt_stamped_target.header.frame_id = target_frame;
    pt_stamped_target.header.stamp = time;

    return true;
}

bool isGoodHelper(const PersonTrajectory& pt,
                  const geometry_msgs::PointStamped& robot_loc,
                  const bool robot_localized)
{
    // Blacklist
    if (hs_.checkBlacklist(pt.track_id))
        return false;

    // Occlusion
    bool is_occluded = true;
    const int smallest_idx = std::max(
            int(0), int(pt.trajectory.size() - number_past_track_for_occlusion_));
    for (int i = pt.trajectory.size() - 1; i >= smallest_idx; --i)
    {
        is_occluded = is_occluded && pt.trajectory.at(i).is_occluded;
    }
    if (is_occluded)
        return false;

    // If there is no robot localization, cannot performance distance and
    // rechability check.
    if (!robot_localized)
        return true;

    // Distance
    const geometry_msgs::Point& person_loc_point = pt.trajectory.back().pose.pose.position;
    const double dist = sqrt(pow(person_loc_point.x - robot_loc.point.x, 2.0)
                             + pow(person_loc_point.y - robot_loc.point.y, 2.0)
                             + pow(person_loc_point.z - robot_loc.point.z, 2.0));
    if (dist > max_dist_)
        return false;

    // Obstacle
    geometry_msgs::PointStamped person_loc;
    person_loc.header = robot_loc.header;
    person_loc.point = person_loc_point;
    const bool is_reachable = checkReachability(robot_loc, person_loc);
    if (!is_reachable)
        return false;

    return true;
}

int selectHelper(const std::vector<int>& trajectory_indices,
                 std::vector<bool>& helper_flag)
{
    const std::vector<PersonTrajectory>& trajectories = person_trajectories_.trajectories;
    helper_flag.clear();
    helper_flag.reserve(trajectory_indices.size());

    // Localize robot
    geometry_msgs::PointStamped robot_loc;
    const bool robot_localized = getRobotLocation(robot_loc);

    // Loop through all tracked trajectories, and check if the person is a good helper.
    int selected_idx = -1;
    int idx_of_last_selected_person = -1;
    for (const int idx : trajectory_indices)
    {
        const PersonTrajectory& person = trajectories.at(idx);
        const bool can_help = isGoodHelper(person, robot_loc, robot_localized);
        helper_flag.push_back(can_help);
        if (can_help)
        {
            selected_idx = idx;
            if (hs_.last_selected_person_id == person.track_id)
            {
                idx_of_last_selected_person = idx;
            }
        }
    }

    // If last person is still a valid choice, keep choosing him.
    if (hs_.keep && idx_of_last_selected_person >= 0)
    {
        return idx_of_last_selected_person;
    }

    // Otherwise simply use one of the potential helper.
    return selected_idx;
}

void callback(const TrackedPersons::ConstPtr &tps)
{
    // Update trajectories
    std::vector<int> trajectory_indices;
    trajectory_indices.reserve(tps->tracks.size());
    for (int i = 0; i < tps->tracks.size(); i++)
    {
        const int idx = updateTrajectories(tps->tracks.at(i));
        trajectory_indices.push_back(idx);
    }
    person_trajectories_.header = tps->header;

    // Publish trajectories
    person_trajectories_pub_.publish(person_trajectories_);

    if (!hs_.active)
        return;

    // Select helper
    std::vector<bool> helper_flag;
    const int selected_person_idx = selectHelper(trajectory_indices, helper_flag);

    // No good helper
    if (selected_person_idx < 0)
    {
        hs_.publishSuccess(false);
        hs_.publishStatus();
        return;
    }

    // Publish good helpers (trajectories and visualization markers)
    DetectedPersons good_helpers_viz;
    good_helpers_viz.header = tps->header;
    PersonTrajectories good_helpers;
    good_helpers.header = tps->header;
    for (int i = 0; i < tps->tracks.size(); i++)
    {
        if (helper_flag.at(i))
        {
            DetectedPerson helper_viz;
            helper_viz.confidence = 0.5;
            helper_viz.pose.pose.position = tps->tracks.at(i).pose.pose.position;
            good_helpers_viz.detections.push_back(helper_viz);
            good_helpers.trajectories.push_back(
                    person_trajectories_.trajectories.at(trajectory_indices.at(i)));
        }
    }
    good_helpers_pub_.publish(good_helpers);
    good_helpers_viz_pub_.publish(good_helpers_viz);

    // Publish selected helper (trajectory and visualization marker)
    PersonTrajectory selected_person_trajectory =
            person_trajectories_.trajectories.at(selected_person_idx);
    selected_helper_pub_.publish(selected_person_trajectory);
    DetectedPersons selected_person_viz;
    selected_person_viz.header = tps->header;
    DetectedPerson s_viz;
    s_viz.confidence = 1.0;
    s_viz.pose.pose.position = selected_person_trajectory.trajectory.back().pose.pose.position;
    selected_person_viz.detections.push_back(s_viz);
    selected_helper_viz_pub_.publish(selected_person_viz);

    // Publish status and success
    hs_.publishStatus();
    hs_.last_selected_person_id = selected_person_trajectory.track_id;
    hs_.publishSuccess(true);
}

// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(message_filters::Subscriber<TrackedPersons> &sub_tra){
    if(!person_trajectories_pub_.getNumSubscribers()
        && !selected_helper_pub_.getNumSubscribers()
        && !selected_helper_viz_pub_.getNumSubscribers()
        && !good_helpers_pub_.getNumSubscribers()
        && !good_helpers_viz_pub_.getNumSubscribers()
        && !hs_.status_pub.getNumSubscribers()
    ) {
        ROS_DEBUG("Trajectories: No subscribers. Unsubscribing.");
        sub_tra.unsubscribe();
    } else {
        ROS_DEBUG("Trajectories: New subscribers. Subscribing.");
        sub_tra.subscribe();
    }
}

void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
{
    // if (!map_ptr_)
    //     ROS_INFO("No map.");
    map_ptr_ = std::make_shared<nav_msgs::OccupancyGrid>(*msg);
    // if (map_ptr_)
    //     ROS_INFO("Received map.");
}

int main(int argc, char **argv)
{
    //init ROS
    ros::init(argc, argv, "pedestrian_trajectories");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    std::string sub_topic_tracked_persons;
    std::string sub_topic_new_search;
    std::string sub_topic_reset_helper_blacklist;
    std::string sub_topic_stop_helper_selection;
    std::string pub_topic_trajectories;
    std::string pub_topic_selected_helper;
    std::string pub_topic_potential_helpers;
    std::string pub_topic_selected_helper_vis;
    std::string pub_topic_potential_helpers_vis;
    std::string pub_topic_helper_search_status;
    std::string pub_topic_helper_selected;

    tf_ = new tf::TransformListener();

    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("tracked_persons", sub_topic_tracked_persons, std::string("/rwth_tracker/tracked_persons"));
    private_node_handle_.param("get_new_helper", sub_topic_new_search, std::string("/rwth_tracker/get_new_helper"));
    private_node_handle_.param("reset_helper_blacklist", sub_topic_reset_helper_blacklist, std::string("/rwth_tracker/reset_helper_blacklist"));
    private_node_handle_.param("stop_helper_selection", sub_topic_stop_helper_selection, std::string("/rwth_tracker/stop_helper_selection"));
    private_node_handle_.param("camera_frame", camera_frame_, std::string("/camera/"));
    // helper selection options
    private_node_handle_.param("keep", hs_.keep, true);
    private_node_handle_.param("strict", hs_.strict, false);
    // private_node_handle_.param("remember", hs_.remember, true);
    //threshold to reidentify helper
    // private_node_handle_.param("helper_reid_thresh", helper_reid_thresh, double(50));

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

    // Listen to map
    std::string sub_topic_map;
    private_node_handle_.param("maps", sub_topic_map, std::string("/map_latched"));
    ros::Subscriber sub_map = n.subscribe(sub_topic_map, 1, mapCallback);

    // Create a topic publisher
    private_node_handle_.param("person_trajectories", pub_topic_trajectories, std::string("/rwth_tracker/person_trajectories"));
    private_node_handle_.param("selected_helper", pub_topic_selected_helper, std::string("/rwth_tracker/selected_helper"));
    private_node_handle_.param("potential_helpers", pub_topic_potential_helpers, std::string("/rwth_tracker/potential_helpers"));
    private_node_handle_.param("selected_helper_vis", pub_topic_selected_helper_vis, std::string("/rwth_tracker/selected_helper_vis"));
    private_node_handle_.param("potential_helpers_vis", pub_topic_potential_helpers_vis, std::string("/rwth_tracker/potential_helpers_vis"));
    private_node_handle_.param("helper_search_status", pub_topic_helper_search_status, std::string("/rwth_tracker/helper_search_status"));
    private_node_handle_.param("helper_selected", pub_topic_helper_selected, std::string("/rwth_tracker/helper_selected"));
    person_trajectories_pub_ = n.advertise<PersonTrajectories>(pub_topic_trajectories, 1, con_cb, con_cb);
    selected_helper_pub_ = n.advertise<PersonTrajectory>(pub_topic_selected_helper, 1, con_cb, con_cb);
    good_helpers_pub_ = n.advertise<PersonTrajectories>(pub_topic_potential_helpers, 1, con_cb, con_cb);
    selected_helper_viz_pub_ = n.advertise<DetectedPersons>(pub_topic_selected_helper_vis, 1, con_cb, con_cb);
    good_helpers_viz_pub_ = n.advertise<DetectedPersons>(pub_topic_potential_helpers_vis, 1, con_cb, con_cb);

    hs_.status_pub = n.advertise<std_msgs::Bool>(pub_topic_helper_search_status, 1, con_cb, con_cb);
    hs_.stop_ack_pub = n.advertise<std_msgs::Bool>(sub_topic_stop_helper_selection + "_ACK", 1, con_cb, con_cb);
    hs_.start_ack_pub = n.advertise<std_msgs::Bool>(sub_topic_new_search + "_ACK", 1, con_cb, con_cb);
    hs_.success_pub = n.advertise<std_msgs::Bool>(pub_topic_helper_selected, 1, con_cb, con_cb);

    ros::spin();

    return 0;
}
