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
//#include "argsParser.h"
#include "configs.h"
#include "YoloLayer.h"
#include "dataReader.h"
#include "eval.h"
#include <chrono>

using namespace darknet_ros_msgs;
using namespace message_filters;
using namespace std;
//using namespace argsParser;
using namespace Tn;
using namespace Yolo;

ros::Publisher pub_boundingboxes;

vector<float> prepareImage(cv::Mat& img)
{
    using namespace cv;

//    int c = parser::getIntValue("C");
//    int h = parser::getIntValue("H");   //net h
//    int w = parser::getIntValue("W");   //net w

    int h = 608;
    int w = 608; //hardcode .... what this mean?
    int c = 3;

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
    auto t_start = chrono::high_resolution_clock::now();

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

    auto t_end = chrono::high_resolution_clock::now();
    float total = chrono::duration<float, milli>(t_end - t_start).count();
    cout << "Time taken for nms is " << total << " ms." << endl;
}


vector<Bbox> postProcessImg(cv::Mat& img,vector<Detection>& detections,int classes)
{
    using namespace cv;

//    int h = parser::getIntValue("H");   //net h
//    int w = parser::getIntValue("W");   //net w

    int h = 608;
    int w = 608;

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
    float nmsThresh = 0.45; //hardcode now
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

vector<string> split(const string& str, char delim)
{
    stringstream ss(str);
    string token;
    vector<string> container;
    while (getline(ss, token, delim)) {
        container.push_back(token);
    }

    return container;
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
//        it.subscribe();
//            }
//}
trtNet* net_ptr;
int outputCount;
unique_ptr<float[]> outputData;

void Callback(const sensor_msgs::ImageConstPtr& img)
{
    ROS_INFO("get image");
    outputData = unique_ptr<float[]>(new float[outputCount]);
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
    ROS_INFO("inputData size is %d", inputData.size());
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

    cout<<"boxes number"<<boxes.size()<<endl;
        for(const auto& item : boxes)
        {
//            if(item.score<0.1)
//                continue;
            cv::rectangle(cvmat,cv::Point(item.left,item.top),cv::Point(item.right,item.bot),cv::Scalar(0,0,255),3,8,0);
            cout << "class=" << item.classId << " prob=" << item.score*100 << endl;
            cout << "left=" << item.left << " right=" << item.right << " top=" << item.top << " bot=" << item.bot << endl;
        }
        cv::imshow("result",cvmat);
        cv::waitKey(25);

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


    ROS_DEBUG("yoloconvertor: Queue size for synchronisation is set to: %i", queue_size);

    // Image transport handle
    image_transport::ImageTransport it(private_node_handle_);
    ros::Subscriber sub_message; //Subscribers have to be defined out of the if scope to have affect.
    image_transport::Subscriber subscriber_img = it.subscribe(image_topic.c_str(),queue_size,Callback);

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
    private_node_handle_.param("bounding_boxs", boundingboxes, string("/bounding_boxs"));
    pub_boundingboxes = n.advertise<darknet_ros_msgs::BoundingBoxes>(boundingboxes, 10);/* con_cb, con_cb)*/;

    //build engine
//    string saveName = "../tensorRT_yolo/yolov3_fp16.engine";  //hardcode
    net_ptr = new trtNet(engine_path);
    outputCount = net_ptr->getOutputSize()/sizeof(float);

    ros::spin();

    return 0;
}
