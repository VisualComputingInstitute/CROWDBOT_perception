#ifndef MAP_H
#define MAP_H
#include <ros/time.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <ros/ros.h>
#include <stdio.h>

using namespace std;

class mymap{

public:
    mymap(ros::NodeHandle n, string map_topic)
    {
        listener = new tf::TransformListener();
        mapname_ = map_topic;
        ros::Subscriber map_sub_ = n.subscribe(mapname_, 1, &mymap::map_callback, this);
        ROS_INFO("Waiting for the map");
    }


    // this call back only get called once, and get the map.
    void map_callback(const nav_msgs::OccupancyGridConstPtr& ogptr)
    {
        oc_map = *ogptr;
    }

    void update_transform_camera2frame(string camera_frame_id, ros::Time detected_time){
        string map_frame_id = oc_map.header.frame_id;
        try {
            listener->waitForTransform(map_frame_id, camera_frame_id, detected_time, ros::Duration(1.0));
            listener->lookupTransform(map_frame_id, camera_frame_id, detected_time, camera2map);  //from camera to map
        }
        catch (tf::TransformException ex){
           ROS_WARN_THROTTLE(20.0, "Failed transform lookup from camera frame to map frame. The map data is empty:%s", oc_map.data.empty() ? "true" : "false", ex.what());
        }
    }


    bool is_pos_occupied(tf::Vector3 pos3D_incam)
    {
        // 1. transform this pos3D to map frame;
        tf::Vector3 pos3D_inmap = camera2map*pos3D_incam;
        // 2. get the index for the map. Since the map's left corner (0,0 cordinate in map image) is not exactly the origin in map frame, it is biasd.
        double map_resolution = oc_map.info.resolution;
        tf::Vector3 map_origin(oc_map.info.origin.position.x, oc_map.info.origin.position.y, 0.0);  // this is in meter unit
        //oc_map.info.origin.orientation;  // we assume no rotation bias, since most systm ignore this map rotation
        pos3D_inmap -= map_origin;
        unsigned int map_ind_x = static_cast<unsigned int>(pos3D_inmap.x()*1.0/map_resolution);
        unsigned int map_ind_y = static_cast<unsigned int>(pos3D_inmap.y()*1.0/map_resolution);

        //3. see if it is occupanied in map
        bool result;
        auto grid_value = oc_map.data[map_ind_x+map_ind_y*oc_map.info.width]; //row-major
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
    nav_msgs::OccupancyGrid oc_map;
    std::string mapname_;
    ros::Subscriber map_sub_;
    tf::StampedTransform camera2map;
    tf::TransformListener* listener;
};

#endif // MAP_H
