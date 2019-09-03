// ROS includes.
#include <ros/ros.h>
#include <iostream>
#include <cmath>
#include <ros/time.h>
#include <image_transport/subscriber_filter.h>
#include <image_transport/image_transport.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <string.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <darknet_ros_msgs/BoundingBox.h>
#include <darknet_ros_msgs/BoundingBoxes.h>

#include "Vector.h"
#include "Visual.h"

#include "TrtNet.h"
#include "configs.h"
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"

using namespace darknet_ros_msgs;
using namespace message_filters;
using namespace std;
using namespace Tn;
using namespace Yolo;

ros::Publisher pub_boundingboxes;
image_transport::Publisher pub_result_image;
float g_detect_threshold;
float g_nms_threshold;

// yolo network parameter, should be fixed with the given engine file?
int g_net_h = 608;
int g_net_w = 608;
int g_net_c = 3;

vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

//    int c = parser::getIntValue("C");
//    int h = parser::getIntValue("H");   //net h
//    int w = parser::getIntValue("W");   //net w

    int h = g_net_h;
    int w = g_net_w; //hardcode .... what this mean?
    int c = g_net_c;

    float scale = min(float(w)/img.cols,float(h)/img.rows);
    auto scaleSize = cv::Size(img.cols * scale,img.rows * scale);

    cv::Mat rgb ;
    cv::cvtColor(img, rgb, CV_BGR2RGB);
    cv::Mat resized;
    cv::resize(rgb, resized,scaleSize,0,0,INTER_CUBIC);

    cv::Mat cropped(h, w,CV_8UC3, 127);
    Rect rect((w- scaleSize.width)/2, (h-scaleSize.height)/2, scaleSize.width,scaleSize.height);
    resized.copyTo(cropped(rect));

    cv::Mat img_float;
    if (c == 3)
        cropped.convertTo(img_float, CV_32FC3, 1/255.0);
    else
        cropped.convertTo(img_float, CV_32FC1 ,1/255.0);

    //HWC TO CHW
    vector<Mat> input_channels(c);
    cv::split(img_float, input_channels);

    vector<float> result(h*w*c);
    auto data = result.data();
    int channelLength = h * w;
    for (int i = 0; i < c; ++i) {
        memcpy(data,input_channels[i].data,channelLength*sizeof(float));
        data += channelLength;
    }

    return result;
}

void DoNms(vector<Detection>& detections,int classes ,float nmsThresh)
{
    vector<vector<Detection>> resClass;
    resClass.resize(classes);

    for (const auto& item : detections)
        resClass[item.classId].push_back(item);

    auto iouCompute = [](float * lbox, float* rbox)
    {
        float interBox[] = {
            max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    };

    vector<Detection> result;
    for (int i = 0;i<classes;++i)
    {
        auto& dets =resClass[i];
        if(dets.size() == 0)
            continue;

        sort(dets.begin(),dets.end(),[=](const Detection& left,const Detection& right){
            return left.prob > right.prob;
        });

        for (unsigned int m = 0;m < dets.size() ; ++m)
        {
            auto& item = dets[m];
            result.push_back(item);
            for(unsigned int n = m + 1;n < dets.size() ; ++n)
            {
                if (iouCompute(item.bbox,dets[n].bbox) > nmsThresh)
                {
                    dets.erase(dets.begin()+n);
                    --n;
                }
            }
        }
    }

    //swap(detections,result);
    detections = move(result);
}


vector<Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

//    int h = parser::getIntValue("H");   //net h
//    int w = parser::getIntValue("W");   //net w

    int h = g_net_h;
    int w = g_net_w;

    //scale bbox to img
    int width = img.cols;
    int height = img.rows;
    float scale = min(float(w)/width,float(h)/height);
    float scaleSize[] = {width * scale,height * scale};

    //correct box
    for (auto& item : detections)
    {
        auto& bbox = item.bbox;
        bbox[0] = (bbox[0] * w - (w - scaleSize[0])/2.f) / scaleSize[0];
        bbox[1] = (bbox[1] * h - (h - scaleSize[1])/2.f) / scaleSize[1];
        bbox[2] /= scaleSize[0];
        bbox[3] /= scaleSize[1];
    }

    //nms
    //float nmsThresh = parser::getFloatValue("nms");
    float nmsThresh = g_nms_threshold; //hardcode now
    if(nmsThresh > 0)
        DoNms(detections,classes,nmsThresh);

    vector<Bbox> boxes;
    for(const auto& item : detections)
    {
        auto& b = item.bbox;
        Bbox bbox =
        {
            item.classId,   //classId
            max(int((b[0]-b[2]/2.)*width),0), //left
            min(int((b[0]+b[2]/2.)*width),width), //right
            max(int((b[1]-b[3]/2.)*height),0), //top
            min(int((b[1]+b[3]/2.)*height),height), //bot
            item.prob       //score
        };
        boxes.push_back(bbox);
    }

    return boxes;
}






//void connectCallback(ros::Subscriber &sub_msg,
//                     image_transport::Subscriber &sub_img,
//                     image_transport::ImageTransport &it){
//    if(!pub_boundingboxes.getNumSubscribers()) {
//        ROS_DEBUG("yoloconvertor: No subscribers. Unsubscribing.");
//        sub_msg.shutdown();
//        sub_img.unsubscribe();

//    } else {
//        ROS_DEBUG("yoloconvertor: New subscribers. Subscribing.");
//        sub_img.subscribe(it,sub_img.getTopic().c_str(),1);
//            }
//}
trtNet* net_ptr;
int outputCount;

void Callback(const sensor_msgs::ImageConstPtr& img)
{
    if(pub_boundingboxes.getNumSubscribers()==0&&pub_result_image.getNumSubscribers()==0)
        return;  // no one subscribe me, just do nothing.
    unique_ptr<float[]> outputData(new float[outputCount]);
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
    cv::Mat cvmat = cv_ptr->image;
    vector<float> inputData = prepareImage(cvmat);
    net_ptr->doInference(inputData.data(), outputData.get());

    //Get Output
    auto output = outputData.get();

    //first detect count
    int count = output[0];
    //later detect result
    vector<Detection> result;
    result.resize(count);
    memcpy(result.data(), &output[1], count*sizeof(Detection));
    int classNum = 80;
    auto boxes = postProcessImg(cvmat,result,classNum);  // Ithink here we get boxes.

     // generate darknet bounding box
     darknet_ros_msgs::BoundingBoxes bbs;
     bbs.image_header = img->header;
     bbs.header = img->header;
     bbs.header.frame_id = "detection";
     bbs.header.stamp = ros::Time::now();
     bbs.image_header.stamp = ros::Time::now();
     for(const auto& item: boxes)
     {
         if(item.classId == 0 && item.score > g_detect_threshold) // classid=0 is pedstrain, threshold hardcode
         {
            darknet_ros_msgs::BoundingBox box;
            box.Class =std::string("person");
            box.probability = item.score;
            box.xmax = item.right;
            box.xmin = item.left;
            box.ymax = item.bot;
            box.ymin = item.top;
            bbs.bounding_boxes.push_back(box);
         }
     }

     // generate darkent detection image

     /*for(auto it = bbs.bounding_boxes.begin();it!=bbs.bounding_boxes.end();++it)
     {
         float height = it->ymax - it->ymin;
         float width = it->xmax - it->xmin;
         float x =(float)(it->xmin>0?it->xmin:0);// make sure x and y are in the image.
         float y = (float)(it->ymin>0?it->ymin:0);
         render_bbox_2D(x,y,width,height,cvmat,0);
         render_text(it->Class,cvmat,x,y,0);
     }*/
     // publish image
     //sensor_msgs::ImagePtr msg = cv_bridge::CvImage(img->header, "bgr8", cvmat).toImageMsg();
     //pub_result_image.publish(msg);
     // publish boundingbox
     pub_boundingboxes.publish(bbs);

}


int main(int argc, char **argv)
{
    // Set up ROS.
    ros::init(argc, argv, "tensorrt_yolo");
    ros::NodeHandle n;

    // Declare variables that can be modified by launch file or command line.
    int queue_size;
    string image_topic;
    string boundingboxes;
    string engine_path;


    // Initialize node parameters from launch file or command line.
    // Use a private node handle so that multiple instances of the node can be run simultaneously
    // while using different parameters.
    ros::NodeHandle private_node_handle_("~");
    private_node_handle_.param("queue_size", queue_size, int(10));
    private_node_handle_.param("image", image_topic, string("/hardware/video/valeo/rectificationNIKRLeft/PanoramaImage"));
    private_node_handle_.param("bounding_boxes", boundingboxes, string("darknet_ros/bounding_boxes"));
    private_node_handle_.param("engine_path", engine_path, string("oops I need engine!"));
    private_node_handle_.param("detect_threshold",g_detect_threshold, float(0.7));
    private_node_handle_.param("nms_threshold", g_nms_threshold, float(0.45));

    ROS_DEBUG("yoloconvertor: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    image_transport::Subscriber subscriber_img = it.subscribe(image_topic.c_str(),1,Callback);

//    ros::SubscriberStatusCallback con_cb = boost::bind(&connectCallback,
//                                                       boost::ref(sub_message),
//                                                       boost::ref(subscriber_img),
//                                                       boost::ref(it));



//    //The real queue size for synchronisation is set here.
//    sync_policies::ApproximateTime<Image> MySyncPolicy(queue_size);

//    const sync_policies::ApproximateTime< Image> MyConstSyncPolicy = MySyncPolicy;
//    Synchronizer< sync_policies::ApproximateTime<Image> > sync(MyConstSyncPolicy,
//                                                               subscriber_img);

//    // Decide which call back should be used.
//    if(strcmp(ground_plane.c_str(), "") == 0) {
//        ROS_FATAL("ground_plane: need ground_plane to produce 3d information");
//    } else {
//        sync.registerCallback(boost::bind(&Callback, _1));
//    }

    // Create publishers
    private_node_handle_.param("bounding_boxes", boundingboxes, string("/bounding_boxes"));
    pub_boundingboxes = n.advertise<darknet_ros_msgs::BoundingBoxes>(boundingboxes, 1);/* con_cb, con_cb)*/;
    //debug image publisher
    string pub_topic_result_image;
    private_node_handle_.param("tensorRT_yolo_out_image", pub_topic_result_image, string("/tensorRT_yolo_image"));
    pub_result_image = it.advertise(pub_topic_result_image.c_str(), 1);  // con_cb maybe...
    //build engine
//    string saveName = "../tensorRT_yolo/yolov3_fp16.engine";  //hardcode
    net_ptr = new trtNet(engine_path);
    outputCount = net_ptr->getOutputSize()/sizeof(float);

    ros::spin();

    return 0;
}
