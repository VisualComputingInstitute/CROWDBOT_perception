#include "ros/ros.h"
#include "std_msgs/String.h"

#include "frame_msgs/TrackedPersons.h"

ros::Publisher pub_person_trajectories;

void callback(const TrackedPersons::ConstPtr &tp) {
  ROS_INFO("I heard: [%s]", tp.header.frame.c_str());
}

// Connection callback that unsubscribes from the tracker if no one is
// subscribed.
void connectCallback(message_filters::Subscriber<TrackedPersons> &sub_tra) {
  if (!pub_person_trajectories.getNumSubscribers()) {
    ROS_DEBUG("Trajectories: No subscribers. Unsubscribing.");
    sub_tra.unsubscribe();
  } else {
    ROS_DEBUG("Trajectories: New subscribers. Subscribing.");
    sub_tra.subscribe();
  }
}

int main(int argc, char **argv) {
  // init ROS
  ros::init(argc, argv, "listener");
  ros::NodeHandle n;

  // Declare variables that can be modified by launch file or command line.
  int queue_size;
  string sub_topic_tracked_persons;
  string pub_topic_trajectories;

  // Initialize node parameters from launch file or command line.
  // Use a private node handle so that multiple instances of the node can be run
  // simultaneously while using different parameters.
  ros::NodeHandle private_node_handle_("~");
  private_node_handle_.param("queue_size", queue_size, int(10));

  ROS_DEBUG(
      "pedestrian_trajectories: Queue size for synchronisation is set to: %i",
      queue_size);

  // Create a subscriber.
  ros::Subscriber sub = n.subscribe("chatter", 1000, callback);

  // Set queue size to 1 because generating a queue here will only pile up
  // images and delay the output by the amount of queued images The immediate
  // unsubscribe is necessary to start without subscribing to any topic because
  // message_filters does nor allow to do it another way.
  message_filters::Subscriber<TrackedPersons> subscriber_tracks(
      n, sub_topic_tracked_persons.c_str(), 1);
  subscriber_tracks.unsubscribe();

  ros::SubscriberStatusCallback con_cb =
      boost::bind(&connectCallback, boost::ref(subscriber_tracks));

  // Create a topic publisher
  private_node_handle_.param("person_trajectories", pub_topic_trajectories,
                             string("/rwth_tracker/person_trajectories"));
  pub_tracked_persons = n.advertise<frame_msgs::PersonTrajectories>(
      pub_topic_trajectories, 10, con_cb, con_cb);

  ros::spin();

  return 0;
}
