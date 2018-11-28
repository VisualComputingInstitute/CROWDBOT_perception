
#ifndef _NENG_VISUAL_H
#define	_NENG_VISUAL_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string.h>

void render_bbox_2D(float x_float,float y_float, float width, float height, cv::Mat& image,
                    int r, int g, int b)
{
    cv::Point2f pt1(x_float,y_float);
    cv::Point2f pt2(x_float + width,y_float + height);
    cv::rectangle(image,pt1,pt2,cv::Scalar(r,g,b),3);
}


void render_text(const string& text, cv::Mat& image, float x_float, float y_float, int r, int g, int b )
{
    int font_face = cv::FONT_HERSHEY_TRIPLEX;
    double font_scale = 1;
    int thickness = 2;    
    cv::putText(image, text, cv::Point2f(x_float,y_float),font_face,font_scale,cv::Scalar(r,g,b), thickness, 8, false);
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


