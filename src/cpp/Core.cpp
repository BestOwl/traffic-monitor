//
// Created by hao on 2020/12/17.
//

#include "Core.h"
#include <opencv2/imgproc.hpp>

using namespace cv;

void DrawRect(cv::Mat& img, DetectedObject obj)
{
    Point topLeft(obj.bbox.xMin, obj.bbox.yMin);
    Point bottomRight(obj.bbox.xMax, obj.bbox.yMax);
    Scalar color;
    switch (obj.classId) {
        case 0:
            color = Scalar(5, 250, 90);
            break;
        case 1:
            color = Scalar(5, 250, 235);
            break;
        case 2:
            color = Scalar(250, 250, 5);
            break;
        case 3:
            color = Scalar(140, 5, 250);
            break;
    }
    rectangle(img, topLeft, bottomRight, color);
}