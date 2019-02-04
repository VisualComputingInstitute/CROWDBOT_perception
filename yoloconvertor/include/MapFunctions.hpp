#ifndef MAP_H
#define MAP_H
#include <ros/time.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>

using namespace std;

class MapFunctions{

public:
    MapFunctions(ros::NodeHandle n, string map_topic)
    {
        listener_ = new tf::TransformListener();
        mapname_ = map_topic;
        map_sub_ = n.subscribe(mapname_, 1, &MapFunctions::mapCallback, this);
        ROS_INFO("initialzie the map class");
    }


    // this call back only get called once, and get the map.
    void mapCallback(const nav_msgs::OccupancyGridConstPtr& ogptr)
    {
        oc_map_ = *ogptr;
        ROS_INFO("get map!");

    }

    void updateCamera2frameTransform(string camera_frame_id, ros::Time detected_time){
        string map_frame_id = oc_map_.header.frame_id;
        try {
            listener_->waitForTransform(map_frame_id, camera_frame_id, detected_time, ros::Duration(1.0));
            listener_->lookupTransform(map_frame_id, camera_frame_id, detected_time, camera2map_);  //from camera to map
        }
        catch (tf::TransformException ex){
           ROS_WARN_THROTTLE(20.0, "Failed transform lookup from camera frame to map frame. The map data is empty:%s", oc_map_.data.empty() ? "true" : "false", ex.what());
           return;
        }
    }


    bool isPosOccupied(tf::Vector3 pos3D_incam)
    {
        if(oc_map_.data.empty())
        {
            ROS_WARN_THROTTLE(20.0, "The map data is empty:%s. Please check the map topic", oc_map_.data.empty() ? "true" : "false");
            return false;
        }
        // 1. transform this pos3D to map frame;
        tf::Vector3 pos3D_inmap = camera2map_*pos3D_incam;
        // 2. get the index for the map. Since the map's left corner (0,0 cordinate in map image) is not exactly the origin in map frame, it is biasd.
        double map_resolution = oc_map_.info.resolution;
        tf::Vector3 map_origin(oc_map_.info.origin.position.x, oc_map_.info.origin.position.y, 0.0);  // this is in meter unit
        //oc_map_.info.origin.orientation;  // we assume no rotation bias, since most systm ignore this map rotation
        pos3D_inmap -= map_origin;
        unsigned int map_ind_x = static_cast<unsigned int>(pos3D_inmap.x()*1.0/map_resolution);
        unsigned int map_ind_y = static_cast<unsigned int>(pos3D_inmap.y()*1.0/map_resolution);

        //3. see if it is occupanied in map
        bool result;
        auto grid_value = oc_map_.data[map_ind_x+map_ind_y*oc_map_.info.width]; //row-major
        if(grid_value == 100) // now with the map from map_serve, 100 is occupied
        {
            result = true;   // do not pass the check, directly go to the next bounding box.
        }
        else
        {
            result = false;
        }
        return result;
    }

private:
    nav_msgs::OccupancyGrid oc_map_;
    std::string mapname_;
    ros::Subscriber map_sub_;
    tf::StampedTransform camera2map_;
    tf::TransformListener* listener_;
};

#endif // MAP_H
