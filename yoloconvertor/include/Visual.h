
#ifndef _NENG_VISUAL_H
#define	_NENG_VISUAL_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

#include <iostream>
using namespace std;


using cv::Scalar;



///////////////// color we used to visual track id/////////////////////////////////////////////////////////////////
std::vector<cv::Scalar> color_vec{Scalar(230, 25, 75), Scalar(60, 180, 75), Scalar(255, 225, 25),
                                  Scalar(0, 130, 200), Scalar(245, 130, 48), Scalar(145, 30, 180),
                                  Scalar(70, 240, 240), Scalar(240, 50, 230), Scalar(210, 245, 60),
                                  Scalar(250, 190, 190), Scalar(0, 128, 128), Scalar(230, 190, 255),
                                  Scalar(170, 110, 40), Scalar(255, 250, 200), Scalar(128, 0, 0),
                                  Scalar(170, 255, 195), Scalar(128, 128, 0), Scalar(255, 215, 180),
                                  Scalar(0, 0, 128), Scalar(128, 128, 128), Scalar(255, 255, 255), Scalar(0, 0, 0)
};
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

bool is_point_invalid(cv::Point2f pt, cv::Mat& image)
{
    if(pt.x<0 || pt.y<0 || pt.x>(image.cols-1) || pt.y>(image.rows-1))
            return true;
    else
        return false;
}

void render_traj_point(double x, double y, cv::Mat& image, unsigned int track_id, unsigned int frame_idx)
{
    cv::Point2f pt(x,y);
    const int max_radius = static_cast<int>(image.cols/80.0);
    const int radius_step = static_cast<int>(radius_step/20.0);
    const int min_radius = 2;
    int radius = max_radius - frame_idx*radius_step;
    radius = std::max(radius,min_radius);
    //  frame_idx = 0 is the latest frame, and we draw points in later frame big.
    size_t color_num = color_vec.size();
    cv::circle(image,pt,radius,color_vec[track_id%color_num],-1); // thickness = -1, means we fill this circle
}



void render_traj_line(double x0, double y0,double x1, double y1,cv::Mat& image, unsigned int track_id)
{
    cv::Point2f pt0(x0,y0);
    cv::Point2f pt1(x1,y1);
    size_t color_num = color_vec.size();
    if(is_point_invalid(pt0,image)||is_point_invalid(pt1,image))
        return;  // point we give may not inside our image
    int line_thick = static_cast<int>(image.cols/250.0);
    cv::line(image,pt0,pt1,color_vec[track_id%color_num],line_thick);
}



void render_bbox_2D(float x_float,float y_float, float width, float height, cv::Mat& image,
                    int r, int g, int b)
{
    cv::Point2f pt1(x_float,y_float);
    cv::Point2f pt2(x_float + width,y_float + height);
    cv::rectangle(image,pt1,pt2,cv::Scalar(r,g,b),3);
}

void render_bbox_2D(float x_float,float y_float, float width, float height, cv::Mat& image,
                                       unsigned int track_id)
{
    cv::Point2f pt1(x_float,y_float);
    cv::Point2f pt2(x_float + width,y_float + height);
    cv::rectangle(image,pt1, pt2, color_vec[track_id% color_vec.size()],3);
}



void render_text(const string& text, cv::Mat& image, float x_float, float y_float, int r, int g, int b )
{
    int font_face = cv::FONT_HERSHEY_TRIPLEX;
    double font_scale = 1;
    int thickness = 2;
    cv::putText(image, text, cv::Point2f(x_float,y_float),font_face,font_scale,cv::Scalar(r,g,b), thickness, 8, false);
}

void render_text(const string& text, cv::Mat& image, float x_float, float y_float, unsigned int track_id )
{
    int font_face = cv::FONT_HERSHEY_TRIPLEX;
    double font_scale = 1;
    int thickness = 2;
    cv::putText(image, text, cv::Point2f(x_float,y_float),font_face,font_scale,color_vec[track_id % color_vec.size()], thickness, 8, false);
}


std::string float_to_string(double num)
{
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << num;
    return ss.str();
}



std::string int_to_string(int num)
{
    std::stringstream ss;
    ss << std::fixed << num;
    return ss.str();
}






#endif	/* _NENG_VISUAL_H */

