#include "ros/ros.h"
#include "ros/time.h"
#include "spencer_tracking_msgs/TrackedPersons.h"
#include "spencer_tracking_msgs/TrackedPersons2d.h"

#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <message_filters/subscriber.h>

#include <iostream>
#include <fstream>

using namespace std;
using namespace spencer_tracking_msgs;
using namespace message_filters;

uint64_t currentIndex = 1;

void callback(const TrackedPersonsConstPtr& eval3d,
              const TrackedPersons2dConstPtr& eval2d,
              const TrackedPersonsConstPtr& track3d,
              const TrackedPersons2dConstPtr& track2d)
{

    // Confidence (not used)
    const int conf = -1;
    char safe_string_char[128];
    static string path_to_eval = "/home/stefan/results/spencer_tracker/groundtruth.txt";
    static string path_to_tracks = "/home/stefan/results/spencer_tracker/tracker_result.txt";
    ofstream aStream;

    // Evaluation 3d
    for(TrackedPerson eva_3d : eval3d->tracks)
    {
        // Evaluation 2d
        for(TrackedPerson2d eva_2d : eval2d->boxes)
        {
            if(eva_2d.track_id == eva_3d.track_id){
                std::cout << currentIndex << ", " << eva_2d.track_id << ", " << eva_2d.x << ", " << eva_2d.y << ", " << eva_2d.w << ", " << eva_2d.h << ", "
                        << conf << ", " << eva_3d.pose.pose.position.x << ", " << eva_3d.pose.pose.position.y << ", 0" << std::endl;
                sprintf(safe_string_char, path_to_eval.c_str());
                aStream.open(safe_string_char, std::ios_base::app);
                aStream << currentIndex << ", " << eva_2d.track_id << ", " << eva_2d.x << ", " << eva_2d.y << ", " << eva_2d.w << ", " << eva_2d.h << ", "
                        << conf << ", " << eva_3d.pose.pose.position.x << ", " << eva_3d.pose.pose.position.y << ", 0" << std::endl;
                aStream.close();
                break;
            }
        }

    }

    // Trackingresult 3d
    for(TrackedPerson track_3d : track3d->tracks)
    {
        // Trackingresult 2d
        for(TrackedPerson2d track_2d : track2d->boxes)
        {
            if(track_2d.track_id == track_3d.track_id){
                std::cout << currentIndex << ", " << track_2d.track_id << ", " << track_2d.x << ", " << track_2d.y << ", " << track_2d.w << ", " << track_2d.h << ", "
                          << conf << ", " << track_3d.pose.pose.position.x << ", " << track_3d.pose.pose.position.y << ", 0" << std::endl;
                sprintf(safe_string_char, path_to_tracks.c_str());
                aStream.open(safe_string_char, std::ios_base::app);
                aStream << currentIndex << ", " << track_2d.track_id << ", " << track_2d.x << ", " << track_2d.y << ", " << track_2d.w << ", " << track_2d.h << ", "
                        << conf << ", " << track_3d.pose.pose.position.x << ", " << track_3d.pose.pose.position.y << ", 0" << std::endl;
                aStream.close();
                break;
            }
        }
    }

    ++currentIndex;
}

/*
// Connection callback that unsubscribes from the tracker if no one is subscribed.
void connectCallback(Subscriber<TrackedPersons> &eval3d,
                     Subscriber<TrackedPersons2d> &eval2d,
                     Subscriber<TrackedPersons> &track3d,
                     Subscriber<TrackedPersons2d> &track2d)
{
    if(!pub_annotated_persons.getNumSubscribers())
    {
        ROS_DEBUG("Annotations: No subscribers. Unsubscribing.");
        sub_cam.unsubscribe();
        sub_gp.unsubscribe();
        sub_vo.unsubscribe();
    }
    else
    {
        ROS_DEBUG("Annotations: New subscribers. Subscribing.");
        sub_cam.subscribe();
        sub_gp.subscribe();
        sub_vo.subscribe();
    }
}

}
*/

int main(int argc, char **argv)
{
  using namespace std;

  ros::init(argc, argv, "annotationreceiver");
  ros::NodeHandle n;
  
  ///////////////////////////////////////////////////////////////////////////////////
  // Parameters
  int queue_size;
  string topic_eval3d;
  string topic_eval2d;
  string topic_track3d;
  string topic_track2d;

  ros::NodeHandle private_node_handle_("~");
  private_node_handle_.param("queue_size", queue_size, int(10));
  private_node_handle_.param("annotated_persons", topic_eval3d, string("/spencer/perception/annotated_persons"));
  private_node_handle_.param("annotated_persons2d", topic_eval2d, string("/spencer/perception/annotated_persons2d"));
  private_node_handle_.param("tracked_persons", topic_track3d, string("/spencer/perception/tracked_persons"));
  private_node_handle_.param("tracked_persons_2d", topic_track2d, string("/spencer/perception/tracked_persons_2d"));


  message_filters::Subscriber<TrackedPersons> subscriber_eval3d(n, topic_eval3d.c_str(), 1); subscriber_eval3d.subscribe();
  message_filters::Subscriber<TrackedPersons2d> subscriber_eval2d(n, topic_eval2d.c_str(), 1); subscriber_eval2d.subscribe();
  message_filters::Subscriber<TrackedPersons> subscriber_track3d(n, topic_track3d.c_str(), 1); subscriber_track3d.subscribe();
  message_filters::Subscriber<TrackedPersons2d> subscriber_track2d(n, topic_track2d.c_str(), 1); subscriber_track2d.subscribe();

  /*ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
                                                     boost::ref(subscriber_eval3d),
                                                     boost::ref(subscriber_eval2d),
                                                     boost::ref(subscriber_track3d),
                                                     boost::ref(subscriber_track2d));*/

  // Register Callback
  const sync_policies::ApproximateTime<TrackedPersons, TrackedPersons2d, TrackedPersons, TrackedPersons2d> MyConstSyncPolicy(queue_size); //The real queue size for synchronisation is set here.
  Synchronizer< sync_policies::ApproximateTime<TrackedPersons, TrackedPersons2d, TrackedPersons, TrackedPersons2d> >
  sync(MyConstSyncPolicy, subscriber_eval3d, subscriber_eval2d, subscriber_track3d, subscriber_track2d);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));


  ros::spin();

  return 0;
}
