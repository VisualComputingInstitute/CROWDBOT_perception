#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <std_msgs/Int8.h>
#include <fstream>
#include <string>
#include <iostream>
#include <message_filters/subscriber.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/time_synchronizer.h>
#include <rwth_perception_people_msgs/UpperBodyDetector.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <QImage>
#include <QPainter>
#include <vector>
#include "rwth_upper_body_skeleton_detector/GetUpperBodySkeleton.h"


unsigned counter = 0;
unsigned detection = 0;

image_transport::Publisher pub_result_image;
ros::ServiceClient upper_body_skeleton_client;

int joint_connections[9] = {-1,0,1,1,1,3,4,5,6};
unsigned bone_colors[6][3] = {{0,0,255},{0,0,255},{255,0,255},{255,0,0},{0,255,0},{255,255,0}};

void convert_to_2d(rwth_upper_body_skeleton_detector::GetUpperBodySkeleton &srv, std::vector<float> &_2d_positions)
{
		
	
      //  char filepath[200];
      //  sprintf(filepath,"data/detections/detection_%d_%d.txt",counter,++detection);
      //  std::ofstream myfile;
      //  myfile.open (filepath);
    	 
        unsigned _2d_counter = 0;
    	for (unsigned j = 0 ; j < 27 ; j = j + 3)
	 {
		unsigned short c1 = 480/2;
	    	unsigned short c2 = 640/2;
		unsigned short row = (525 * srv.response.upper_body_skeleton_joint_positions[j])/srv.response.upper_body_skeleton_joint_positions[j+2] + c2;
		unsigned short col = (-525 * srv.response.upper_body_skeleton_joint_positions[j+1])/srv.response.upper_body_skeleton_joint_positions[j+2] + c1;
		// myfile << srv.response.upper_body_skeleton_joint_positions[j];
	//	 myfile << "," << srv.response.upper_body_skeleton_joint_positions[j+1];
        //         myfile << "," << srv.response.upper_body_skeleton_joint_positions[j+2];
        //         myfile <<"\n";
		_2d_positions[_2d_counter++] = row;
		_2d_positions[_2d_counter++] = col;
                //myfile << row << "," << col << "\n";
	}
//	myfile.close();
 
}

void preprocessing(cv_bridge:: CvImageConstPtr cv_ptr, float *bounding_box,
                   rwth_upper_body_skeleton_detector::GetUpperBodySkeleton & srv)
{ 
    unsigned short start_col,start_row,width,height;
    if ((bounding_box[0]-5) >= 0)
    	start_col = bounding_box[0]-5;
    else
        start_col = bounding_box[0];
    if ((bounding_box[1]-5) >= 0) 
    	 start_row = bounding_box[1]-5;
    else
	start_row = bounding_box[1];
    if ((start_col + bounding_box[2]+5) <= 639)
    	 width = start_col + bounding_box[2]+5;
    else
        width = start_col + bounding_box[2];
    
    float median_depth = bounding_box[4];
   
    if ((start_row + bounding_box[3] + 200/median_depth) <= 479)
    	height = start_row + bounding_box[3] + 200/median_depth;
    else
        height = start_row + bounding_box[3];
  
   
 //  unsigned total_cols = width - start_col + 1;
 //  unsigned total_rows = height - start_row + 1; 	   
 //  pixels_x.resize(total_cols*total_rows);
 //  pixels_y.resize(total_cols*total_rows);
 //  depths.resize(total_cols*total_rows);
   
   unsigned  pixel_counter = 1;
   //unsigned long col_counter = 0;
  // unsigned long depth_counter = 0;

   //std::cout << "totalcols: "<<total_cols << ", total rows : " << total_rows << "total depths : " << total_cols * total_rows << "\n"; 
   

    for (unsigned row = start_row+1; row <  height; row++)
    {
        for (unsigned col = start_col+1; col < width ; col++)
        {
              if (cv_ptr->image.at<float>(row,col) <= (median_depth + .6) && cv_ptr->image.at<float>(row,col) > 0)

                 {
                      float depth = cv_ptr->image.at<float>(row,col);
                      srv.request.depths[0] = srv.request.depths[0] + 1;
                      srv.request.pixels_x[0] = srv.request.pixels_x[0] + 1;
                      srv.request.pixels_y[0] = srv.request.pixels_y[0] + 1; 
                      srv.request.depths[pixel_counter] = depth;
                      srv.request.pixels_x[pixel_counter] = col+1;
                      srv.request.pixels_y[pixel_counter] = row+1;
                     // depths.push_back(depth);
                     // pixels_y.push_back(row+1);
                     // pixels_x.push_back(col+1);
                      unsigned long linearindex = 480*((col+1) - 1)+  (row+1);
                      srv.request.depth_image[(linearindex-1)] = depth;
 		      pixel_counter++;
		      
               }
	    /*else
	    	{
		      depths[counte] = 0;
                      pixels_x[counter] = 0;
                      pixels_y[counter] = 0;

	        }*/
          
          //  std:: cout << depth_counter << "," << row_counter << "," << col_counter  << "\n";    
        }
     }
   
   
}



 void render_bbox_with_skeleton(QImage& image,
                    int r, int g, int b, int lineWidth,
                    std::vector<std::vector<float> > &skeletons,
                    const rwth_perception_people_msgs ::UpperBodyDetector::ConstPtr &detections,
		    unsigned total_skeletons_predicted)
{

    QPainter painter(&image);
    QColor qColor;
    qColor.setRgb(r, g, b);
    QPen pen;
    pen.setColor(qColor);
    pen.setWidth(5);

    painter.setPen(pen);
    
    unsigned N = total_skeletons_predicted; 
    unsigned joint_counter = 0;

    for(unsigned i = 0; i < N; i++){
        joint_counter = 0;
        int x =(int) detections->pos_x[i]-5;
        int y =(int) detections->pos_y[i]-5;
        int w =(int) detections->width[i]+5;
        int h =(int) detections->height[i]+200/detections->median_depth[i];

        painter.drawLine(x,y, x+w,y);
        painter.drawLine(x,y, x,y+h);
        painter.drawLine(x+w,y, x+w,y+h);
        painter.drawLine(x,y+h, x+w,y+h);
        unsigned bone_counter = 0;
        for (unsigned j = 0 ; j < 14 ; j = j + 2)
		{
                  
		  if (joint_connections[joint_counter] >= 0)
		   { unsigned short index = joint_connections[joint_counter];	
                     qColor.setRgb(bone_colors[bone_counter][0], bone_colors[bone_counter][1], bone_colors[bone_counter][2]);
    		     pen.setColor(qColor);
                     painter.setPen(pen);
		     painter.drawLine(skeletons[i][index*2],skeletons[i][index*2+1], skeletons[i][j],skeletons[i][j+1]);
		     bone_counter++;	
		   }
                joint_counter++;
		
	    }
	}

}
     
    	

void callback(const sensor_msgs::ImageConstPtr& depth , const sensor_msgs::ImageConstPtr& color, const rwth_perception_people_msgs ::UpperBodyDetector::ConstPtr &upper)
{
   
    // reading the depth image from sensor in open_cv
    detection = 0;
    cv_bridge:: CvImageConstPtr cv_ptr;
    cv_ptr = cv_bridge::toCvShare(depth);
    cv::Size s = cv_ptr->image.size();
    int rows = s.height;
    int cols = s.width;
    
    
    ROS_INFO("Recieved frame %d", counter++);
    
    
   
   unsigned short total_upper_bodies = upper->pos_x.size();

   std::vector<std::vector<float> > _2d_upper_body_skeletons(total_upper_bodies,std::vector<float>(18));    
   //float _2d_upper_body_skeletons[total_upper_bodies][18];   
   unsigned total_skeletons_predicted = 0;

   //ROS_INFO("N : %d",total_upper_bodies);
   //std::cout << "\n";
   for (unsigned i =0 ; i < total_upper_bodies ; i++)
   	{ 
            if (upper->median_depth[i] <= 1.5)
            		{ ROS_INFO("Person too close."); continue;}
            rwth_upper_body_skeleton_detector::GetUpperBodySkeleton srv;
            
    	    size_t N = rows*cols;
            //srv.request.depth_image.resize(N); 
    	    for (unsigned row = 0; row < N ; row++)
    			srv.request.depth_image[row] = 0;
            //std::vector<float> tmp_skeleton(27);
            //std::vector<float> bounding_box(5);
            float bounding_box[5];    
            bounding_box[0] = upper->pos_x[i]; bounding_box[1] = upper->pos_y[i]; 
            bounding_box[2] = upper->width[i]; bounding_box[3] = upper->height[i];
            bounding_box[4] = upper->median_depth[i];

            srv.request.pixels_x[0] = 0;
	    srv.request.pixels_y[0] = 0;
	    srv.request.depths[0] = 0;

	    
            preprocessing(cv_ptr,bounding_box,srv);
            if (srv.request.depths.size() < 5)
                  continue;
            if (upper_body_skeleton_client.call(srv))
		{
                     // std::vector<float> _2d_skeleton;
                      convert_to_2d(srv,_2d_upper_body_skeletons[i]);
		      total_skeletons_predicted++;
                    //_2d_upper_body_skeletons.push_back(_2d_skeleton);	
		}
	    else
		ROS_INFO("Failed to calll upper_body_skeleton_server");
        }
   
   QImage image_rgb(&color->data[0], color->width, color->height, QImage::Format_RGB888);
   render_bbox_with_skeleton(image_rgb, 0, 0, 255, 2,_2d_upper_body_skeletons,upper,total_skeletons_predicted);
   sensor_msgs::Image sensor_image;
   sensor_image.header = color->header;
   sensor_image.height = image_rgb.height();
   sensor_image.width  = image_rgb.width();
   sensor_image.step   = color->step;
   std :: vector<unsigned char> image_bits(image_rgb.bits(), image_rgb.bits()+sensor_image.height*sensor_image.width*3);
   sensor_image.data = image_bits;
   sensor_image.encoding = color->encoding;
   pub_result_image.publish(sensor_image);
  /* try
  {
    //writing the image to file
    counter++;
    
    char filepath[200];
    sprintf(filepath,"data/images/image_%d.txt",counter);
    ROS_INFO("Recieved image");
    std::ofstream myfile;
    myfile.open (filepath);
    for (unsigned row = 0; row < rows; row++)
     {
	for(unsigned col = 0; col < cols; col++)
  		myfile <<cv_ptr->image.at<float>(row,col) <<"\t" ;
	myfile << "\n";
      }
    myfile.close();
   //writing bounding box to file */
   //size_t ndetections = upper->pos_x.size();
   //upper_body_skeleton_detector :: GetUpperBodySkeleton srv;
   /*for (unsigned i = 0; i < ndetections; i++ )
	{
                
	        unsigned start_x = upper->pos_x[i], start_y = upper->pos_y[i], width = upper->width[i],height = upper->height[i];
                float median_depth = upper->median_depth[i];
   		char detectionpath[200];
    		sprintf(detectionpath,"data/detections/detection_%d.txt",counter);
    		std::ofstream myfile2;
    		myfile2.open (detectionpath);
		myfile2 << start_x << "\n" << start_y <<"\n" << width << "\n" << height << "\n" <<median_depth;
    		myfile2.close();
	}
    
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("Could not write image to file");
  }
  /*int rows,cols;
  
  cv_ptr = cv_bridge::toCvShare(msg);
  */
}

int main(int argc, char **argv)
{
  
  ros::init(argc, argv, "upper_body_skeleton_client_node");
  
  ros::NodeHandle n;
  upper_body_skeleton_client = n.serviceClient<rwth_upper_body_skeleton_detector::GetUpperBodySkeleton>("/rwth_upper_body_skeleton_detector/get_upper_body_skeleton");
  image_transport::ImageTransport it(n);
 	
  ROS_INFO("Starting upper_body_skeleton_client....");

  std ::string depth_image_msg = "/spencer/sensors/rgbd_front_top/depth/image";
  std ::string rgb_image_msg = "/spencer/sensors/rgbd_front_top/rgb/image_raw";
  std ::string upper_body_msg = "/spencer/perception_internal/people_detection/rgbd_front_top/upper_body_detector/detections";

  image_transport::SubscriberFilter subscriber_depthimage(it,depth_image_msg, 1);
  image_transport::SubscriberFilter subscriber_rgbimage(it,rgb_image_msg,1);
  message_filters::Subscriber<rwth_perception_people_msgs::UpperBodyDetector> subscriber_upperbody(n,upper_body_msg, 1);
  typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image,rwth_perception_people_msgs::UpperBodyDetector> MySyncPolicy;
 //const SyncType sync_policy(10);
  message_filters::Synchronizer< MySyncPolicy > mysync(MySyncPolicy( 10 ), subscriber_depthimage,subscriber_rgbimage ,subscriber_upperbody);
  mysync.registerCallback(boost::bind(&callback, _1, _2,_3));

  //Publisher
  pub_result_image = it.advertise("/rwth_upper_body_skeleton/colorimage", 1);

  ros::spin();

  return 0;
}
